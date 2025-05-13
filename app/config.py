import json
import threading
import tomllib
import yaml # Added for YAML support
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


class LLMSettings(BaseModel):
    model: str = Field(..., description="Model name")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(..., description="API key")
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")
    max_input_tokens: Optional[int] = Field(
        None,
        description="Maximum input tokens to use across all requests (None for unlimited)",
    )
    temperature: float = Field(1.0, description="Sampling temperature")
    api_type: str = Field(..., description="Azure, Openai, or Ollama")
    api_version: str = Field(..., description="Azure Openai version if AzureOpenai")


class ProxySettings(BaseModel):
    server: str = Field(None, description="Proxy server address")
    username: Optional[str] = Field(None, description="Proxy username")
    password: Optional[str] = Field(None, description="Proxy password")


class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="Search engine the llm to use")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"],
        description="Fallback search engines to try if the primary engine fails",
    )
    retry_delay: int = Field(
        default=60,
        description="Seconds to wait before retrying all engines again after they all fail",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of times to retry all engines when all fail",
    )
    lang: str = Field(
        default="en",
        description="Language code for search results (e.g., en, zh, fr)",
    )
    country: str = Field(
        default="us",
        description="Country code for search results (e.g., us, cn, uk)",
    )


class BrowserSettings(BaseModel):
    headless: bool = Field(False, description="Whether to run browser in headless mode")
    disable_security: bool = Field(
        True, description="Disable browser security features"
    )
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )
    chrome_instance_path: Optional[str] = Field(
        None, description="Path to a Chrome instance to use"
    )
    wss_url: Optional[str] = Field(
        None, description="Connect to a browser instance via WebSocket"
    )
    cdp_url: Optional[str] = Field(
        None, description="Connect to a browser instance via CDP"
    )
    proxy: Optional[ProxySettings] = Field(
        None, description="Proxy settings for the browser"
    )
    max_content_length: int = Field(
        2000, description="Maximum length for content retrieval operations"
    )


class SandboxSettings(BaseModel):
    """Configuration for the execution sandbox"""

    use_sandbox: bool = Field(False, description="Whether to use the sandbox")
    image: str = Field("python:3.12-slim", description="Base image") # Specification used 3.11, existing was 3.12
    work_dir: str = Field("/workspace", description="Container working directory")
    memory_limit: str = Field("1g", description="Memory limit") # Specification used 1g, existing was 512m
    cpu_limit: float = Field(1.5, description="CPU limit") # Specification used 1.5, existing was 1.0
    timeout: int = Field(600, description="Default command timeout (seconds)") # Specification used 600, existing was 300
    network_enabled: bool = Field(
        True, description="Whether network access is allowed" # Specification used true, existing was false
    )

class KnowledgeStoreSettings(BaseModel):
    embedding_model: str = Field("all-MiniLM-L6-v2")
    vector_db_type: str = Field("faiss_IndexFlatL2")
    persist_path: str = Field("workspace/knowledge_store")

class WebInterfaceSettings(BaseModel):
    enabled: bool = Field(True)
    host: str = Field("0.0.0.0")
    port: int = Field(8501)
    api_base_url: str = Field("http://localhost:8000")

class AgentSettings(BaseModel):
    name: str = Field("OpenManusAgent")
    max_steps: int = Field(50)
    dual_model_system: Optional[Dict[str, str]] = Field(default_factory=lambda: {"primary_model_key": "deepseek-v3", "secondary_model_key": "deepseek-r1"})

class SupervisionSettings(BaseModel):
    default_autonomy_level: str = Field("supervised")
    checkpoint_triggers: List[Dict[str, Any]] = Field(default_factory=list)
    intervention_timeout_seconds: int = Field(300)

class LoggingSettings(BaseModel):
    level: str = Field("INFO")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = None

class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server"""

    type: str = Field(..., description="Server connection type (sse or stdio)")
    url: Optional[str] = Field(None, description="Server URL for SSE connections")
    command: Optional[str] = Field(None, description="Command for stdio connections")
    args: List[str] = Field(
        default_factory=list, description="Arguments for stdio command"
    )


class MCPSettings(BaseModel):
    """Configuration for MCP (Model Context Protocol)"""

    server_reference: str = Field(
        "app.mcp.server", description="Module reference for the MCP server"
    )
    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="MCP server configurations"
    )

    @classmethod
    def load_server_config(cls) -> Dict[str, MCPServerConfig]:
        """Load MCP server configuration from JSON file"""
        config_path = PROJECT_ROOT / "config" / "mcp.json"

        try:
            config_file = config_path if config_path.exists() else None
            if not config_file:
                return {}

            with config_file.open() as f:
                data = json.load(f)
                servers = {}

                for server_id, server_config in data.get("mcpServers", {}).items():
                    servers[server_id] = MCPServerConfig(
                        type=server_config["type"],
                        url=server_config.get("url"),
                        command=server_config.get("command"),
                        args=server_config.get("args", []),
                    )
                return servers
        except Exception as e:
            raise ValueError(f"Failed to load MCP server config: {e}")


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    agent: Optional[AgentSettings] = None
    supervision: Optional[SupervisionSettings] = None
    knowledge_store: Optional[KnowledgeStoreSettings] = None
    web_interface: Optional[WebInterfaceSettings] = None
    sandbox: Optional[SandboxSettings] = None
    browser_config: Optional[BrowserSettings] = None
    search_config: Optional[SearchSettings] = None
    mcp_config: Optional[MCPSettings] = None
    logging: Optional[LoggingSettings] = None

    class Config:
        arbitrary_types_allowed = True


class Config:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config_data = None # Store raw loaded data
                    self._app_config: Optional[AppConfig] = None # Store parsed AppConfig
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path_and_type() -> tuple[Optional[Path], Optional[str]]:
        root = PROJECT_ROOT
        config_dir = root / "config"
        
        # Priority: YAML, then TOML
        yaml_path = config_dir / "config.yaml"
        if yaml_path.exists():
            return yaml_path, "yaml"
        
        example_yaml_path = config_dir / "config.example.yaml"
        if example_yaml_path.exists():
            return example_yaml_path, "yaml"

        toml_path = config_dir / "config.toml"
        if toml_path.exists():
            return toml_path, "toml"
        
        example_toml_path = config_dir / "config.example.toml"
        if example_toml_path.exists():
            return example_toml_path, "toml"
            
        return None, None

    def _load_config_data(self) -> dict:
        config_path, config_type = self._get_config_path_and_type()
        
        if not config_path or not config_type:
            raise FileNotFoundError("No configuration file (config.yaml, config.example.yaml, config.toml, or config.example.toml) found in config directory")

        with config_path.open("r" if config_type == "yaml" else "rb") as f:
            if config_type == "yaml":
                try:
                    return yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML configuration file {config_path}: {e}")
            elif config_type == "toml":
                try:
                    return tomllib.load(f)
                except tomllib.TOMLDecodeError as e:
                    raise ValueError(f"Error parsing TOML configuration file {config_path}: {e}")
        return {}

    def _load_initial_config(self):
        self._config_data = self._load_config_data()
        raw_config = self._config_data

        # LLM settings processing (remains similar, assumes dict structure from YAML/TOML)
        base_llm_config = raw_config.get("llm", {}).get("default", {})
        llm_overrides = {k: v for k, v in raw_config.get("llm", {}).items() if k != "default" and isinstance(v, dict)}

        # Ensure default LLM settings are fully populated
        default_llm_settings = LLMSettings(
            model=base_llm_config.get("model", "default_model_not_set"),
            base_url=base_llm_config.get("base_url", "default_base_url_not_set"),
            api_key=base_llm_config.get("api_key", "default_api_key_not_set"),
            max_tokens=base_llm_config.get("max_tokens", 4096),
            max_input_tokens=base_llm_config.get("max_input_tokens"),
            temperature=base_llm_config.get("temperature", 0.7),
            api_type=base_llm_config.get("api_type", "openai"),
            api_version=base_llm_config.get("api_version", "v1")
        )

        parsed_llm_configs = {"default": default_llm_settings}
        for name, override_config_data in llm_overrides.items():
            # Create LLMSettings instance for each override, falling back to defaults if keys are missing
            parsed_llm_configs[name] = LLMSettings(
                model=override_config_data.get("model", default_llm_settings.model),
                base_url=override_config_data.get("base_url", default_llm_settings.base_url),
                api_key=override_config_data.get("api_key", default_llm_settings.api_key),
                max_tokens=override_config_data.get("max_tokens", default_llm_settings.max_tokens),
                max_input_tokens=override_config_data.get("max_input_tokens", default_llm_settings.max_input_tokens),
                temperature=override_config_data.get("temperature", default_llm_settings.temperature),
                api_type=override_config_data.get("api_type", default_llm_settings.api_type),
                api_version=override_config_data.get("api_version", default_llm_settings.api_version)
            )

        # Browser settings
        browser_raw_config = raw_config.get("browser", {})
        browser_settings = None
        if browser_raw_config:
            proxy_raw_config = browser_raw_config.get("proxy", {})
            proxy_settings = ProxySettings(**proxy_raw_config) if proxy_raw_config.get("server") else None
            browser_params = {k: v for k, v in browser_raw_config.items() if k != "proxy"}
            if proxy_settings:
                browser_params["proxy"] = proxy_settings
            if browser_params: # only create if there are params
                 browser_settings = BrowserSettings(**browser_params)

        # Other settings, directly from YAML/TOML structure if keys match Pydantic models
        search_settings = SearchSettings(**raw_config["search"]) if "search" in raw_config else None
        sandbox_settings = SandboxSettings(**raw_config["sandbox"]) if "sandbox" in raw_config else SandboxSettings() # Default if not present
        agent_settings = AgentSettings(**raw_config["agent"]) if "agent" in raw_config else None
        supervision_settings = SupervisionSettings(**raw_config["supervision"]) if "supervision" in raw_config else None
        knowledge_store_settings = KnowledgeStoreSettings(**raw_config["knowledge_store"]) if "knowledge_store" in raw_config else None
        web_interface_settings = WebInterfaceSettings(**raw_config["web_interface"]) if "web_interface" in raw_config else None
        logging_settings = LoggingSettings(**raw_config["logging"]) if "logging" in raw_config else None

        # MCP settings (loading from mcp.json remains)
        mcp_raw_config = raw_config.get("mcp", {})
        mcp_servers = MCPSettings.load_server_config() # This part is separate as per original logic
        mcp_settings = MCPSettings(server_reference=mcp_raw_config.get("server_reference", "app.mcp.server"), servers=mcp_servers)
        
        self._app_config = AppConfig(
            llm=parsed_llm_configs,
            agent=agent_settings,
            supervision=supervision_settings,
            knowledge_store=knowledge_store_settings,
            web_interface=web_interface_settings,
            sandbox=sandbox_settings,
            browser_config=browser_settings,
            search_config=search_settings,
            mcp_config=mcp_settings,
            logging=logging_settings
        )

    # Property getters to access parsed AppConfig fields
    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._app_config.llm

    @property
    def agent(self) -> Optional[AgentSettings]:
        return self._app_config.agent

    @property
    def supervision(self) -> Optional[SupervisionSettings]:
        return self._app_config.supervision

    @property
    def knowledge_store(self) -> Optional[KnowledgeStoreSettings]:
        return self._app_config.knowledge_store

    @property
    def web_interface(self) -> Optional[WebInterfaceSettings]:
        return self._app_config.web_interface

    @property
    def sandbox(self) -> SandboxSettings:
        return self._app_config.sandbox

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        return self._app_config.browser_config

    @property
    def search_config(self) -> Optional[SearchSettings]:
        return self._app_config.search_config

    @property
    def mcp_config(self) -> MCPSettings:
        return self._app_config.mcp_config
    
    @property
    def logging(self) -> Optional[LoggingSettings]:
        return self._app_config.logging

    @property
    def workspace_root(self) -> Path:
        return WORKSPACE_ROOT

    @property
    def root_path(self) -> Path:
        return PROJECT_ROOT

    # Method to get raw config if needed, e.g., for parts not fitting Pydantic models
    def get_raw_config_value(self, key_path: str, default: Any = None) -> Any:
        """Retrieves a value from the raw loaded config data using a dot-separated path."""
        if not self._config_data:
            return default
        keys = key_path.split(".")
        value = self._config_data
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

config = Config()

