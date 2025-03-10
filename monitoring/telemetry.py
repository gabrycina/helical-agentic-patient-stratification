import os
from dotenv import load_dotenv
import logfire
from pydantic import BaseModel
from datetime import datetime

# Load environment variables
load_dotenv()

class ModelExecutionMetrics(BaseModel):
    """Metrics for model execution and biological insights."""
    model_name: str
    input_shape: tuple
    output_shape: tuple
    execution_time: float
    timestamp: datetime = datetime.now()

def setup_monitoring():
    """Initialize Logfire monitoring."""
    logfire.configure(
        token=os.getenv('LOGFIRE_API_KEY'),  # Use token instead of api_key
        console=logfire.ConsoleOptions(
            min_log_level='debug'  # Capture detailed execution info
        )
    )

def monitor_model_execution(func):
    """Decorator to monitor model execution with Logfire."""
    @logfire.instrument(extract_args=False)  # Don't log large data objects
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
       
        with logfire.span(f'Executing {func.__name__}') as span:
            try:
                result = func(*args, **kwargs)
                
                if hasattr(args[0], 'X'):  # Check if first arg is AnnData
                    adata = args[0]
                    metrics = ModelExecutionMetrics(
                        model_name=func.__name__,
                        input_shape=adata.shape,
                        output_shape=result.shape,
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )
                    
                    # Log metrics as attributes
                    for key, value in metrics.model_dump().items():
                        span.set_attribute(key, str(value))
                
                return result
                
            except Exception as e:
                logfire.error('Model execution failed: {error}', error=str(e))
                span.record_exception(e)
                raise
    
    return wrapper 