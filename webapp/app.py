"""Web interface for comparing prompt templates."""

import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from kastenrag.prompts.models import PromptTemplate
from kastenrag.prompts.registry import (
    register_prompt_template, 
    get_prompt_template, 
    list_templates,
    register_default_templates
)
from kastenrag.prompts.config import (
    initialize_templates, 
    save_template_to_file, 
    load_user_templates
)
from kastenrag.chunkers.atomic import AtomicChunker
from kastenrag.llm import (
    get_llm_client, 
    set_llm_client, 
    create_llm_client,
    get_available_providers,
    get_available_models,
    MockLLMClient
)
from kastenrag.utils.logger import LLMInteractionLogger, ChunkerLogger

# Import background processing module
from webapp.runner import (
    run_test_in_background, 
    get_run_status, 
    STATUS_PENDING,
    STATUS_RUNNING, 
    STATUS_COMPLETED, 
    STATUS_FAILED
)

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_testing')

# Initialize LLM client with mock implementation
mock_client = MockLLMClient(model="mock-gpt-4")
set_llm_client(mock_client)
print(f"Initialized mock LLM client: {mock_client.model}")

# Directory for test data and results
DATA_DIR = Path(__file__).parent.parent / "test_inputs"
RESULTS_DIR = Path(__file__).parent.parent / "webapp" / "results"
TEMPLATE_DIR = Path(__file__).parent.parent / "webapp" / "templates_custom"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

# Test run data structure
test_runs = {}


def is_custom_template(template):
    """Check if a template is a custom template (editable) or built-in (read-only).
    
    There are three ways to identify a custom template:
    1. It has a file in the custom templates directory
    2. It has is_custom=True set directly on the template
    3. It has metadata with is_custom=True
    
    Returns:
        bool: True if template is custom/editable, False if built-in/read-only
    """
    # Check if the template already has is_custom=True
    if template.is_custom:
        return True
    
    # Check if metadata indicates it's a custom template
    if template.metadata and template.metadata.get('is_custom'):
        return True
    
    # Check if a file exists for this template in the custom templates directory
    template_path = TEMPLATE_DIR / f"{template.component_type}_{template.task}_{template.name}.yaml"
    is_custom = template_path.exists()
    
    # Return result but don't modify the template object directly
    return is_custom

@app.route('/')
def index():
    """Render the main page."""
    # Make sure we've registered default templates
    register_default_templates()
    
    # Initialize all templates
    initialize_templates()
    load_user_templates(str(TEMPLATE_DIR))
    
    # Get available templates
    first_pass_templates = [t for t in list_templates(component_type="chunker", task="first_pass")]
    second_pass_templates = [t for t in list_templates(component_type="chunker", task="second_pass")]
    metadata_entity_templates = [t for t in list_templates(component_type="chunker", task="metadata_entity")]
    metadata_topic_templates = [t for t in list_templates(component_type="chunker", task="metadata_topic")]
    metadata_relationship_templates = [t for t in list_templates(component_type="chunker", task="metadata_relationship")]
    
    # Sort templates by name
    first_pass_templates.sort(key=lambda t: t.name)
    second_pass_templates.sort(key=lambda t: t.name)
    metadata_entity_templates.sort(key=lambda t: t.name)
    metadata_topic_templates.sort(key=lambda t: t.name)
    metadata_relationship_templates.sort(key=lambda t: t.name)
    
    # Check and set custom flag for each template
    for template in first_pass_templates + second_pass_templates + metadata_entity_templates + metadata_topic_templates + metadata_relationship_templates:
        is_custom = is_custom_template(template)
        if is_custom:
            # Use setattr to update the field without triggering validation
            template.is_custom = True
    
    # Get available test files
    test_files = []
    for file_path in DATA_DIR.glob("*.txt"):
        test_files.append({
            "name": file_path.name,
            "path": str(file_path),
            "size": file_path.stat().st_size,
        })
    
    # Get previous test runs
    run_list = []
    for run_id, run_data in test_runs.items():
        # Get total combinations either from results or progress data
        if "results" in run_data:
            combinations = len(run_data.get("results", []))
        elif "progress" in run_data and "total_combinations" in run_data["progress"]:
            combinations = run_data["progress"]["total_combinations"]
        else:
            combinations = 0
            
        run_list.append({
            "id": run_id,
            "timestamp": run_data.get("timestamp", "Unknown"),
            "input_file": run_data.get("input_file", "Unknown"),
            "combinations": combinations,
            "status": run_data.get("status", STATUS_COMPLETED),
            "progress": run_data.get("progress", None)
        })
    
    # Sort test runs by timestamp (newest first)
    run_list.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    
    # Get available models for the UI
    available_models = get_available_models()
    
    return render_template(
        'index.html',
        first_pass_templates=first_pass_templates,
        second_pass_templates=second_pass_templates,
        metadata_entity_templates=metadata_entity_templates,
        metadata_topic_templates=metadata_topic_templates,
        metadata_relationship_templates=metadata_relationship_templates,
        test_files=test_files,
        test_runs=run_list,
        available_models=available_models
    )


@app.route('/template/<component_type>/<task>/<n>')
def view_template(component_type, task, n):
    """View a specific template."""
    try:
        template = get_prompt_template(component_type, task, n)
        # Check if this is a custom (editable) template
        is_custom = is_custom_template(template)
        if is_custom:
            template.is_custom = True
        return render_template(
            'view_template.html',
            template=template
        )
    except KeyError:
        flash(f"Template not found: {component_type}/{task}/{n}")
        return redirect(url_for('index'))


@app.route('/template/edit/<component_type>/<task>/<n>', methods=['GET', 'POST'])
def edit_template(component_type, task, n):
    """Edit a specific template."""
    try:
        template = get_prompt_template(component_type, task, n)
        
        # Check if this is a custom template (editable)
        is_custom = is_custom_template(template)
        
        # Don't allow editing of built-in templates
        if not is_custom:
            flash(f"Cannot edit built-in template: {n}. Create a copy instead.")
            return redirect(url_for('view_template', component_type=component_type, task=task, n=n))
            
        # Set is_custom flag for the template
        template.is_custom = True
        
        if request.method == 'POST':
            # Update template
            new_template = PromptTemplate(
                name=request.form.get('name', n),
                description=request.form.get('description', template.description),
                version=request.form.get('version', template.version),
                component_type=component_type,
                task=task,
                template=request.form.get('template', template.template),
                tags=request.form.get('tags', '').split(','),
                updated_at=datetime.now()
            )
            
            # Save to file
            template_path = TEMPLATE_DIR / f"{component_type}_{task}_{new_template.name}.yaml"
            save_template_to_file(new_template, str(template_path))
            
            # Register the template
            register_prompt_template(new_template)
            
            flash(f"Template saved: {component_type}/{task}/{new_template.name}")
            return redirect(url_for('index'))
        
        return render_template(
            'edit_template.html',
            template=template,
            component_type=component_type,
            task=task,
            is_new=False,
            is_custom=is_custom
        )
    except KeyError:
        flash(f"Template not found: {component_type}/{task}/{n}")
        return redirect(url_for('index'))


@app.route('/template/new/<component_type>/<task>', methods=['GET', 'POST'])
def new_template(component_type, task):
    """Create a new template."""
    # Get source template if copying
    source_template_name = request.args.get('from')
    source_template = None
    
    if source_template_name:
        try:
            source_template = get_prompt_template(component_type, task, source_template_name)
        except KeyError:
            flash(f"Source template not found: {component_type}/{task}/{source_template_name}")
    
    if request.method == 'POST':
        # Create new template
        new_template = PromptTemplate(
            name=request.form.get('name', 'new_template'),
            description=request.form.get('description', ''),
            version="1.0.0",
            component_type=component_type,
            task=task,
            template=request.form.get('template', ''),
            tags=request.form.get('tags', '').split(','),
            created_at=datetime.now(),
            # Mark as custom template directly
            is_custom=True
        )
        
        # Save to file
        template_path = TEMPLATE_DIR / f"{component_type}_{task}_{new_template.name}.yaml"
        save_template_to_file(new_template, str(template_path))
        
        # Register the template
        register_prompt_template(new_template)
        
        flash(f"Template created: {component_type}/{task}/{new_template.name}")
        return redirect(url_for('index'))
    
    # Create template for the form (empty or copied)
    if source_template:
        # For copying, prefix name with "copy_of_" and keep other fields
        template = {
            "name": f"copy_of_{source_template.name}",
            "description": source_template.description,
            "version": "1.0.0",
            "template": source_template.template,
            "tags": source_template.tags
        }
        page_title = f"Create Copy of {source_template.name}"
    else:
        # For new template, start empty
        template = {
            "name": "",
            "description": "",
            "version": "1.0.0",
            "template": "",
            "tags": []
        }
        page_title = f"New {component_type} {task} Template"
    
    return render_template(
        'edit_template.html',
        template=template,
        component_type=component_type,
        task=task,
        is_new=True,
        page_title=page_title
    )


@app.route('/run_test', methods=['POST'])
def run_test():
    """Start a test in the background with selected templates and input file."""
    # Get input file
    input_file = request.form.get('input_file')
    if not input_file:
        flash("Please select an input file")
        return redirect(url_for('index'))
    
    # Get selected templates
    first_pass_templates = request.form.getlist('first_pass_templates')
    second_pass_templates = request.form.getlist('second_pass_templates')
    
    # Get selected model
    model_id = request.form.get('model', 'mock-gpt-4')
    provider = request.form.get('provider', 'mock')
    
    # Get metadata options - all set to false by default (rule-based approach)
    use_llm_for_metadata = False
    use_llm_for_entities = False
    use_llm_for_topics = False
    use_llm_for_relationships = False
    
    # Get selected metadata templates (allow multiple)
    entity_templates = request.form.getlist('entity_template')
    topic_templates = request.form.getlist('topic_template')
    relationship_templates = request.form.getlist('relationship_template')
    
    # Enable LLM-based approach for metadata types where templates are selected
    if entity_templates:
        use_llm_for_entities = True
    if topic_templates:
        use_llm_for_topics = True
    if relationship_templates:
        use_llm_for_relationships = True
        
    # Set the legacy flag if any of the individual flags are true
    use_llm_for_metadata = use_llm_for_entities or use_llm_for_topics or use_llm_for_relationships
    
    # Default to 'default' template if LLM is enabled but no template selected
    if use_llm_for_entities and not entity_templates:
        entity_templates = ['default']
    if use_llm_for_topics and not topic_templates:
        topic_templates = ['default']
    if use_llm_for_relationships and not relationship_templates:
        relationship_templates = ['default']
    
    # Find model details
    available_models = get_available_models()
    model_info = next((m for m in available_models if m["id"] == model_id), {})
    if model_info:
        provider = model_info.get("provider", provider)
    
    if not first_pass_templates or not second_pass_templates:
        flash("Please select at least one template for each pass")
        return redirect(url_for('index'))
    
    # Create run ID
    run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Read input file
    try:
        with open(input_file, 'r') as f:
            input_text = f.read()
    except Exception as e:
        flash(f"Error reading input file: {e}")
        return redirect(url_for('index'))
    
    # Initialize the test run data with pending status
    test_runs[run_id] = {
        "id": run_id,
        "timestamp": timestamp,
        "input_file": input_file,
        "input_text": input_text,
        "first_pass_templates": first_pass_templates,
        "second_pass_templates": second_pass_templates,
        "status": STATUS_PENDING,
        "model": {
            "id": model_id,
            "provider": provider
        },
        "metadata_options": {
            "use_llm_for_metadata": use_llm_for_metadata,
            "use_llm_for_entities": use_llm_for_entities,
            "use_llm_for_topics": use_llm_for_topics,
            "use_llm_for_relationships": use_llm_for_relationships,
            "entity_templates": entity_templates,
            "topic_templates": topic_templates,
            "relationship_templates": relationship_templates
        },
        "results": []
    }
    
    # Start the test in a background thread
    run_test_in_background(
        run_id=run_id,
        input_file=input_file,
        input_text=input_text,
        first_pass_templates=first_pass_templates,
        second_pass_templates=second_pass_templates,
        model_id=model_id,
        provider=provider,
        use_llm_for_metadata=use_llm_for_metadata,
        use_llm_for_entities=use_llm_for_entities,
        use_llm_for_topics=use_llm_for_topics,
        use_llm_for_relationships=use_llm_for_relationships,
        entity_templates=entity_templates,
        topic_templates=topic_templates,
        relationship_templates=relationship_templates,
        results_dir=RESULTS_DIR,
        test_runs=test_runs
    )
    
    flash(f"Test run started with {len(first_pass_templates) * len(second_pass_templates)} template combinations. You can leave this page and check back later for results.")
    return redirect(url_for('view_run', run_id=run_id))


@app.route('/run/<run_id>')
def view_run(run_id):
    """View a specific test run."""
    if run_id not in test_runs:
        # Try to load from file
        run_file = RESULTS_DIR / f"{run_id}.json"
        if run_file.exists():
            with open(run_file, 'r') as f:
                test_runs[run_id] = json.load(f)
        else:
            flash(f"Test run not found: {run_id}")
            return redirect(url_for('index'))
    
    run_data = test_runs[run_id]
    
    # Get the latest status for in-progress runs
    status = run_data.get("status", STATUS_COMPLETED)
    
    # Determine if we should auto-refresh the page
    auto_refresh = status in [STATUS_PENDING, STATUS_RUNNING]
    
    return render_template(
        'view_run.html',
        run=run_data,
        auto_refresh=auto_refresh,
        status=status
    )


@app.route('/api/run_status/<run_id>')
def api_run_status(run_id):
    """API endpoint to get the status of a run."""
    # Check if run exists in test_runs
    if run_id not in test_runs:
        # Try to load from file
        run_file = RESULTS_DIR / f"{run_id}.json"
        if run_file.exists():
            try:
                with open(run_file, 'r') as f:
                    test_runs[run_id] = json.load(f)
            except Exception as e:
                return jsonify({"error": f"Error loading run data: {e}"}), 500
        else:
            return jsonify({"error": "Run not found"}), 404
    
    # Get the run data
    run_data = test_runs[run_id]
    
    # Extract status information
    status_data = {
        "run_id": run_id,
        "status": run_data.get("status", STATUS_COMPLETED),
        "progress": run_data.get("progress", {
            "total_combinations": 0,
            "completed_combinations": 0,
            "current_combination": None
        }),
        "started_at": run_data.get("started_at"),
        "completed_at": run_data.get("completed_at"),
        "error": run_data.get("error")
    }
    
    return jsonify(status_data)


@app.route('/delete_run/<run_id>', methods=['POST'])
def delete_run(run_id):
    """Delete a specific test run."""
    if run_id in test_runs:
        del test_runs[run_id]
    
    # Delete file if exists
    run_file = RESULTS_DIR / f"{run_id}.json"
    if run_file.exists():
        run_file.unlink()
    
    flash(f"Test run deleted: {run_id}")
    return redirect(url_for('index'))


@app.route('/templates/<template_type>')
def template_management(template_type):
    """Render the template management page for a specific template type."""
    # Make sure we've registered default templates
    register_default_templates()
    
    # Initialize all templates
    initialize_templates()
    load_user_templates(str(TEMPLATE_DIR))
    
    templates = []
    title = ""
    description = ""
    
    if template_type == "first_pass":
        templates = [t for t in list_templates(component_type="chunker", task="first_pass")]
        title = "First Pass Templates"
        description = "Templates used in the first pass of chunking to extract initial facts from text."
    elif template_type == "second_pass":
        templates = [t for t in list_templates(component_type="chunker", task="second_pass")]
        title = "Second Pass Templates"
        description = "Templates used in the second pass of chunking to refine facts and ensure they are self-contained."
    elif template_type == "metadata":
        # Get all metadata template types
        entity_templates = [t for t in list_templates(component_type="chunker", task="metadata_entity")]
        topic_templates = [t for t in list_templates(component_type="chunker", task="metadata_topic")]
        relationship_templates = [t for t in list_templates(component_type="chunker", task="metadata_relationship")]
        
        # Group them by type
        templates = {
            "entity": entity_templates,
            "topic": topic_templates,
            "relationship": relationship_templates
        }
        
        title = "Metadata Templates"
        description = "Templates used for extracting metadata from chunks, including entities, topics, and relationships."
    else:
        # If the template type is not recognized, redirect to the home page
        flash(f"Unknown template type: {template_type}")
        return redirect(url_for('index'))
    
    # Check and set custom flag for each template
    if template_type == "metadata":
        for template_list in templates.values():
            for template in template_list:
                is_custom = is_custom_template(template)
                if is_custom:
                    template.is_custom = True
    else:
        for template in templates:
            is_custom = is_custom_template(template)
            if is_custom:
                template.is_custom = True
    
    return render_template(
        'template_management.html',
        template_type=template_type,
        templates=templates,
        title=title,
        description=description
    )

@app.route('/api/templates')
def api_templates():
    """API endpoint to get all templates."""
    templates = []
    for template in list_templates():
        templates.append({
            "name": template.name,
            "description": template.description,
            "component_type": template.component_type,
            "task": template.task,
            "version": template.version,
            "tags": template.tags
        })
    
    return jsonify(templates)

@app.route('/pipeline-builder')
def pipeline_builder():
    """Render the pipeline builder interface."""
    return render_template('pipeline_builder.html')

@app.route('/api/pipelines', methods=['POST'])
def save_pipeline():
    """Save a pipeline configuration."""
    try:
        pipeline_data = request.json
        if not pipeline_data:
            return jsonify({"error": "No pipeline data provided"}), 400
        
        # Generate a unique ID for the pipeline
        pipeline_id = str(uuid.uuid4())
        
        # Create the pipelines directory if it doesn't exist
        pipeline_dir = Path(__file__).parent / "pipelines"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the pipeline configuration
        pipeline_file = pipeline_dir / f"{pipeline_id}.json"
        with open(pipeline_file, 'w') as f:
            json.dump({
                "id": pipeline_id,
                "name": pipeline_data.get("name", f"Pipeline {pipeline_id[:8]}"),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "config": pipeline_data
            }, f, indent=2)
        
        return jsonify({
            "id": pipeline_id,
            "message": "Pipeline saved successfully"
        })
    except Exception as e:
        return jsonify({"error": f"Error saving pipeline: {str(e)}"}), 500

@app.route('/api/pipelines', methods=['GET'])
def list_pipelines():
    """List all saved pipeline configurations."""
    try:
        pipeline_dir = Path(__file__).parent / "pipelines"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        pipelines = []
        for pipeline_file in pipeline_dir.glob("*.json"):
            try:
                with open(pipeline_file, 'r') as f:
                    pipeline_data = json.load(f)
                    pipelines.append({
                        "id": pipeline_data.get("id"),
                        "name": pipeline_data.get("name"),
                        "created_at": pipeline_data.get("created_at"),
                        "updated_at": pipeline_data.get("updated_at")
                    })
            except Exception as e:
                print(f"Error loading pipeline {pipeline_file}: {e}")
        
        return jsonify(pipelines)
    except Exception as e:
        return jsonify({"error": f"Error listing pipelines: {str(e)}"}), 500

@app.route('/api/pipelines/<pipeline_id>', methods=['GET'])
def get_pipeline(pipeline_id):
    """Get a specific pipeline configuration."""
    try:
        pipeline_file = Path(__file__).parent / "pipelines" / f"{pipeline_id}.json"
        if not pipeline_file.exists():
            return jsonify({"error": "Pipeline not found"}), 404
        
        with open(pipeline_file, 'r') as f:
            pipeline_data = json.load(f)
        
        return jsonify(pipeline_data)
    except Exception as e:
        return jsonify({"error": f"Error getting pipeline: {str(e)}"}), 500

@app.route('/api/pipelines/<pipeline_id>', methods=['DELETE'])
def delete_pipeline(pipeline_id):
    """Delete a specific pipeline configuration."""
    try:
        pipeline_file = Path(__file__).parent / "pipelines" / f"{pipeline_id}.json"
        if not pipeline_file.exists():
            return jsonify({"error": "Pipeline not found"}), 404
        
        pipeline_file.unlink()
        
        return jsonify({"message": "Pipeline deleted successfully"})
    except Exception as e:
        return jsonify({"error": f"Error deleting pipeline: {str(e)}"}), 500

@app.route('/view-pipeline-run/<run_id>')
def view_pipeline_run(run_id):
    """View the results of a pipeline execution."""
    try:
        run_file = Path(__file__).parent / "pipeline_runs" / f"{run_id}.json"
        if not run_file.exists():
            flash("Pipeline run not found")
            return redirect(url_for('pipeline_builder'))
        
        with open(run_file, 'r') as f:
            run_data = json.load(f)
        
        # Get the pipeline data
        pipeline_id = run_data.get("pipeline_id")
        pipeline_file = Path(__file__).parent / "pipelines" / f"{pipeline_id}.json"
        
        pipeline_data = None
        if pipeline_file.exists():
            with open(pipeline_file, 'r') as f:
                pipeline_data = json.load(f)
        
        return render_template(
            'view_pipeline_run.html',
            run=run_data,
            pipeline=pipeline_data
        )
    except Exception as e:
        flash(f"Error viewing pipeline run: {str(e)}")
        return redirect(url_for('pipeline_builder'))

@app.route('/api/pipeline-runs', methods=['GET'])
def list_pipeline_runs():
    """List all pipeline execution runs."""
    try:
        run_dir = Path(__file__).parent / "pipeline_runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        runs = []
        for run_file in run_dir.glob("*.json"):
            try:
                with open(run_file, 'r') as f:
                    run_data = json.load(f)
                    runs.append({
                        "id": run_data.get("id"),
                        "pipeline_id": run_data.get("pipeline_id"),
                        "executed_at": run_data.get("executed_at"),
                        "status": "completed"
                    })
            except Exception as e:
                print(f"Error loading pipeline run {run_file}: {e}")
        
        # Sort by executed_at (newest first)
        runs.sort(key=lambda r: r.get("executed_at", ""), reverse=True)
        
        return jsonify(runs)
    except Exception as e:
        return jsonify({"error": f"Error listing pipeline runs: {str(e)}"}), 500

@app.route('/api/run-pipeline/<pipeline_id>', methods=['POST'])
def run_pipeline(pipeline_id):
    """Run a specific pipeline configuration."""
    try:
        pipeline_file = Path(__file__).parent / "pipelines" / f"{pipeline_id}.json"
        if not pipeline_file.exists():
            return jsonify({"error": "Pipeline not found"}), 404
        
        with open(pipeline_file, 'r') as f:
            pipeline_data = json.load(f)
        
        # Get the pipeline configuration
        pipeline_config = pipeline_data.get("config", {})
        
        # Import the pipeline executor
        from kastenrag.pipeline.pipeline_executor import PipelineExecutor
        
        # Create and initialize the pipeline executor
        executor = PipelineExecutor(pipeline_config)
        if not executor.initialize():
            return jsonify({"error": "Failed to initialize pipeline"}), 500
        
        # Execute the pipeline
        result = executor.execute()
        
        # Generate a run ID
        run_id = str(uuid.uuid4())
        
        # Save the execution result
        run_dir = Path(__file__).parent / "pipeline_runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        run_file = run_dir / f"{run_id}.json"
        with open(run_file, 'w') as f:
            json.dump({
                "id": run_id,
                "pipeline_id": pipeline_id,
                "executed_at": datetime.now().isoformat(),
                "results": result
            }, f, indent=2)
        
        # Return a response with the run ID
        return jsonify({
            "message": "Pipeline execution completed",
            "run_id": run_id,
            "status": "completed",
            "results": result
        })
    except Exception as e:
        return jsonify({"error": f"Error running pipeline: {str(e)}"}), 500


if __name__ == '__main__':
    # Initialize templates
    initialize_templates()
    load_user_templates(str(TEMPLATE_DIR))
    
    # Create pipeline directory if it doesn't exist
    pipeline_dir = Path(__file__).parent / "pipelines"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing test runs
    for run_file in RESULTS_DIR.glob("*.json"):
        try:
            with open(run_file, 'r') as f:
                run_data = json.load(f)
                run_id = run_data.get("id")
                if run_id:
                    test_runs[run_id] = run_data
        except Exception as e:
            print(f"Error loading test run {run_file}: {e}")
    
    # Run the app
    app.run(debug=True, port=5001)