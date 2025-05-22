from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from typing import Dict, List, Any, Optional
import uuid
import json
from datetime import datetime

from app.api.dependencies import get_current_user
from app.db.models import User
from app.logger import logger

router = APIRouter()

# Store for active websocket connections
websocket_clients: Dict[str, WebSocket] = {}

@router.websocket("/connect/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str
):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    # Register client
    if not client_id:
        client_id = f"client_{uuid.uuid4().hex}"
    
    websocket_clients[client_id] = websocket
    logger.info(f"WebSocket client connected: {client_id}")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection_established",
            "timestamp": datetime.utcnow().isoformat(),
            "client_id": client_id,
            "message": "Connected to OpenAgent UI WebSocket"
        })
        
        # Keep connection alive
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Process message (can be used for client-initiated actions)
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from client {client_id}")
    except WebSocketDisconnect:
        # Client disconnected
        logger.info(f"WebSocket client disconnected: {client_id}")
    finally:
        # Unregister client
        if client_id in websocket_clients:
            del websocket_clients[client_id]

async def broadcast_message(
    message_type: str,
    data: Dict[str, Any],
    execution_id: Optional[str] = None
):
    """
    Broadcast a message to all connected websocket clients.
    If execution_id is provided, only clients subscribed to that execution will receive the message.
    """
    message = {
        "type": message_type,
        "timestamp": datetime.utcnow().isoformat(),
        "execution_id": execution_id,
        "data": data
    }
    
    # Send message to all connected clients
    for client_id, websocket in list(websocket_clients.items()):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")
            # Remove client if connection is broken
            if client_id in websocket_clients:
                del websocket_clients[client_id]
