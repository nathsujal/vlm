from pydantic import BaseModel, Field

class Settings(BaseModel):
    timeline_path: str = "data/timeline.json"
    object_report_path: str = "data/object_report.json"
    property_report_path: str = "data/property_report.json"

    preprocessing_output_dir: str = "data/preprocessing_output"
    visualizer_output_dir: str = "data/visualizer_output"
    
settings = Settings()