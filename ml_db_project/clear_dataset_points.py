from shared.db import get_session
from shared.dataset_models import DatasetPoint

s = get_session()
s.query(DatasetPoint).delete()
s.commit()
print("dataset_points cleared")
