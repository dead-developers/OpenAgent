# DeepSeek Multi-Agent Framework Workflow

This document outlines the workflow and implementation approach for a multi-agent system that leverages the complementary strengths of DeepSeek models. The system uses DeepSeek V3 0425 for code-focused execution and DeepSeek-R1 for reasoning and evaluation.

## System Overview

The workflow creates a cooperative multi-agent system where:

1. DeepSeek V3 breaks down complex tasks into manageable subtasks
2. For each subtask, V3 executes while R1 predicts outcomes in parallel
3. The agents compare results, identify differences, and resolve conflicts
4. Results are stored in a vector database for reinforcement learning

## Detailed Workflow

### 1. Task Breakdown

When a user submits a task, DeepSeek V3 breaks it down into sequential subtasks:

```python
# Pseudo-code for task breakdown
def break_down_task(task):
    prompt = f"""
    Break down the following task into smaller, sequential subtasks:
    {task}
    Return a JSON array of subtasks.
    """
    
    response = v3_model.call(prompt)
    subtasks = extract_json(response)
    return subtasks
```

### 2. Parallel Processing

For each subtask, the system runs execution and prediction in parallel:

```python
# Pseudo-code for parallel processing
def process_subtask(subtask):
    # Start two parallel threads
    v3_result = thread_pool.submit(v3_model.execute_task, subtask)
    r1_prediction = thread_pool.submit(r1_model.predict_execution, subtask)
    
    # Get results
    actual_result = v3_result.result()
    predicted_result = r1_prediction.result()
    
    return actual_result, predicted_result
```

### 3. Difference Identification

The system identifies differences between the execution result and prediction:

```python
# Pseudo-code for difference identification
def identify_differences(actual_result, predicted_result):
    prompt = f"""
    Compare these solutions and identify differences:
    Solution 1: {actual_result}
    Solution 2: {predicted_result}
    Return a JSON array of differences.
    """
    
    comparison = r1_model.call(prompt)
    differences = extract_json(comparison)
    
    return differences if differences else None
```

### 4. Conflict Resolution

When differences are found, both models propose resolutions:

```python
# Pseudo-code for resolution proposals
def get_resolutions(subtask, actual_result, predicted_result, differences):
    v3_resolution = v3_model.propose_resolution(
        subtask, actual_result, predicted_result, differences
    )
    
    r1_resolution = r1_model.propose_resolution(
        subtask, actual_result, predicted_result, differences
    )
    
    return v3_resolution, r1_resolution
```

#### 4.1 Resolution Agreement Check

First, check if the models already agree:

```python
def models_agree(v3_resolution, r1_resolution):
    # Normalize whitespace for comparison
    v3_normalized = ' '.join(v3_resolution.split())
    r1_normalized = ' '.join(r1_resolution.split())
    
    return v3_normalized == r1_normalized
```

#### 4.2 Single-Round Argument Presentation

If the models disagree, each presents its case:

```python
def present_arguments(subtask, v3_resolution, r1_resolution):
    v3_argument = v3_model.present_argument(subtask, v3_resolution, r1_resolution)
    r1_argument = r1_model.present_argument(subtask, v3_resolution, r1_resolution)
    
    return v3_argument, r1_argument
```

Each model gets exactly one chance to argue for its solution:

```python
def present_argument(subtask, my_resolution, other_resolution):
    prompt = f"""
    Present a logical argument for why your resolution is better:
    
    Task: {subtask}
    Your resolution: {my_resolution}
    Alternative: {other_resolution}
    
    Focus on:
    1. Task completion
    2. Avoiding unnecessary complexity
    3. Alignment with requirements
    4. Efficiency and elegance
    """
    
    return model.call(prompt)
```

#### 4.3 Concession Opportunity

Each model evaluates the other's argument:

```python
def evaluate_arguments(subtask, v3_argument, r1_argument, v3_resolution, r1_resolution):
    v3_concedes = v3_model.evaluate_argument(
        subtask, r1_argument, v3_resolution, r1_resolution
    )
    
    r1_concedes = r1_model.evaluate_argument(
        subtask, v3_argument, r1_resolution, v3_resolution
    )
    
    return v3_concedes, r1_concedes
```

The evaluation is a straightforward YES/NO decision:

```python
def evaluate_argument(subtask, other_argument, my_resolution, other_resolution):
    prompt = f"""
    Evaluate if this argument is convincing enough to change your mind:
    
    Task: {subtask}
    Your resolution: {my_resolution}
    Their resolution: {other_resolution}
    Their argument: {other_argument}
    
    Respond with only YES or NO.
    """
    
    response = model.call(prompt)
    return "YES" in response.upper()
```

#### 4.4 Hard Decision Criteria

If neither model concedes, apply strict decision rules in order:

```python
def apply_decision_rules(subtask, v3_resolution, r1_resolution):
    # Rule 1: Task Completion (Must fully complete the task)
    v3_completion = evaluate_completion(subtask, v3_resolution)
    r1_completion = evaluate_completion(subtask, r1_resolution)
    
    if v3_completion < 1.0 and r1_completion >= 1.0:
        return r1_resolution
    elif r1_completion < 1.0 and v3_completion >= 1.0:
        return v3_resolution
    elif v3_completion < 1.0 and r1_completion < 1.0:
        # Both fail to complete the task - reject both
        return None
    
    # Rule 2: No Oversolving (Must not add assumed enhancements)
    v3_complexity = evaluate_complexity(subtask, v3_resolution)
    r1_complexity = evaluate_complexity(subtask, r1_resolution)
    
    if v3_complexity > 1.0 and r1_complexity > 1.0:
        # Both oversolved - choose the one with less oversolving
        return v3_resolution if v3_complexity < r1_complexity else r1_resolution
    elif v3_complexity > 1.0:
        return r1_resolution
    elif r1_complexity > 1.0:
        return v3_resolution
    
    # Rule 3: Task Alignment (Choose the most aligned with requirements)
    v3_alignment = evaluate_alignment(subtask, v3_resolution)
    r1_alignment = evaluate_alignment(subtask, r1_resolution)
    
    return v3_resolution if v3_alignment > r1_alignment else r1_resolution
```

### 5. Result Handling

Based on the resolution outcome, the system either:
- Stores the successful resolution with a `.No_rejections` marker
- Stores both failed solutions and restarts the loop
- Moves to the next subtask

```python
def handle_resolution(subtask, resolution, v3_result, r1_prediction, differences):
    if resolution is None:
        # Both solutions rejected - store failures and restart loop
        vector_db.store(
            subtask=subtask,
            v3_result=v3_result,
            r1_prediction=r1_prediction,
            differences=differences,
            resolution=None,
            rejected_solutions=[v3_resolution, r1_resolution]
        )
        
        # Restart loop for this subtask
        return process_subtask_loop(subtask)
    else:
        # Success - store with .No_rejections marker
        vector_db.store(
            subtask=subtask,
            v3_result=v3_result,
            r1_prediction=r1_prediction,
            differences=differences,
            resolution=resolution,
            rejected_solutions=".No_rejections"
        )
        
        return resolution
```

### 6. Vector Database Storage

All results (successes and failures) are stored for learning:

```python
def store_in_vector_db(subtask, v3_result, r1_prediction, 
                      differences=None, resolution=None, 
                      rejected_solutions=None):
    # Create embeddings for semantic search
    subtask_embedding = embedder.encode(subtask)
    result_embedding = embedder.encode(str(v3_result))
    
    # Create record with all relevant information
    record = {
        "subtask": subtask,
        "subtask_embedding": subtask_embedding,
        "v3_result": v3_result,
        "r1_prediction": r1_prediction,
        "differences": differences,
        "resolution": resolution,
        "rejected_solutions": rejected_solutions,
        "timestamp": datetime.now()
    }
    
    # Store in database
    db.insert(record)
```

## Complete Workflow Cycle

The full execution cycle for a single task:

```python
def process_complete_task(task):
    # 1. Break down task
    subtasks = break_down_task(task)
    
    # 2. Process each subtask
    results = []
    for subtask in subtasks:
        # Execute parallel processing
        v3_result, r1_prediction = process_subtask(subtask)
        
        # Identify differences
        differences = identify_differences(v3_result, r1_prediction)
        
        if not differences:
            # No differences - store with .No_rejections marker
            store_in_vector_db(
                subtask, v3_result, r1_prediction, 
                rejected_solutions=".No_rejections"
            )
            results.append(v3_result)
            continue
        
        # Get resolution proposals
        v3_resolution, r1_resolution = get_resolutions(
            subtask, v3_result, r1_prediction, differences
        )
        
        # Check if models already agree
        if models_agree(v3_resolution, r1_resolution):
            store_in_vector_db(
                subtask, v3_result, r1_prediction,
                differences, v3_resolution, ".No_rejections"
            )
            results.append(v3_resolution)
            continue
        
        # Present arguments
        v3_argument, r1_argument = present_arguments(
            subtask, v3_resolution, r1_resolution
        )
        
        # Check for concessions
        v3_concedes, r1_concedes = evaluate_arguments(
            subtask, v3_argument, r1_argument, 
            v3_resolution, r1_resolution
        )
        
        if v3_concedes:
            resolution = r1_resolution
        elif r1_concedes:
            resolution = v3_resolution
        else:
            # Apply hard decision criteria
            resolution = apply_decision_rules(
                subtask, v3_resolution, r1_resolution
            )
        
        # Handle the resolution outcome
        final_result = handle_resolution(
            subtask, resolution, v3_result, r1_prediction, differences
        )
        
        results.append(final_result)
    
    # 3. Synthesize results if needed
    if len(results) > 1:
        final_result = synthesize_results(results, task)
    else:
        final_result = results[0]
    
    return final_result
```

## Key Aspects of the Implementation

1. **Deterministic Resolution**: The conflict resolution process follows strict rules with no randomness or voting.

2. **Single-Chance Arguments**: Each model gets exactly one opportunity to present its case, mirroring a formal debate structure.

3. **Prioritized Decision Rules**: Task completion is the highest priority, followed by avoiding oversolving, then alignment.

4. **Vector Database Learning**: All outcomes (successes and failures) are stored to enable reinforcement learning.

5. **Restart Loop**: If both solutions fail to complete a task, the system restarts on the same subtask until a valid solution is found.

This workflow creates a synergistic system where DeepSeek V3's code strengths are complemented by DeepSeek-R1's reasoning abilities, with clear conflict resolution mechanisms that ensure deterministic outcomes.
