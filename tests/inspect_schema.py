import json
import pydantic
from src.enrichment.schema import EnrichmentOutput, ResearchMetadata

print('pydantic_version:', pydantic.__version__)
print('\n--- EnrichmentOutput.model_json_schema() ---')
print(json.dumps(EnrichmentOutput.model_json_schema(), indent=2, ensure_ascii=False))
# print('\n--- ResearchMetadata.model_json_schema() ---')
# print(json.dumps(ResearchMetadata.model_json_schema(), indent=2, ensure_ascii=False))
