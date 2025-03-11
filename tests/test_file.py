import pytest
from d2draftnet.collect_data1 import DataFetcher
from d2draftnet.config import HERO_MAP

# Fake response for the heroes endpoint.
class FakeResponse:
    def __init__(self, json_data, status_code):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

def fake_get_heroes(url, *args, **kwargs):
    if url.endswith("/heroes"):
        return FakeResponse([
            {"id": 1, "localized_name": "Anti-Mage"},
            {"id": 2, "localized_name": "Axe"},
        ], 200)
    return FakeResponse(None, 404)

def fake_get_pub_data():
    """
    Return fake public match data, including a duplicate match (4001).
    """
    return [
        {
            "match_id": 4001,
            "radiant_team": [1, 2],
            "dire_team": [2, 1],
            "duration": 3600,
            "radiant_win": True,
        },
        {
            "match_id": 4002,
            "radiant_team": [2, 1],
            "dire_team": [1, 2],
            "duration": 4000,
            "radiant_win": False,
        },
        # Duplicate match to test removal.
        {
            "match_id": 4001,
            "radiant_team": [1, 2],
            "dire_team": [2, 1],
            "duration": 3600,
            "radiant_win": True,
        },
    ]

@pytest.fixture
def data_fetcher():
    """
    Create a DataFetcher instance with preset hero mapping and match data.
    This avoids real API calls.
    """
    # Instantiate the DataFetcher (base_url doesn't matter for this test).
    df_instance = DataFetcher("https://api.opendota.com/api")
    # Manually set the hero mapping.
    df_instance.hero_id_to_name = HERO_MAP
    # Preload match_data with two unique matches.
    df_instance.match_data = [
        {
            "match_id": 1001,
            "radiant_draft": ["Anti-Mage", "Axe"],
            "dire_draft": ["Bane", "Crystal Maiden"],
            "duration": 3600,
            "winner": "Radiant",
        },
        {
            "match_id": 1002,
            "radiant_draft": ["Queen of Pain", "Venomancer"],
            "dire_draft": ["Dragon Knight", "Dazzle"],
            "duration": 4000,
            "winner": "Dire",
        },
    ]
    return df_instance

def test_open_dota_id_to_heroname(monkeypatch):
    """
    Test that _openDotaID_2_heroname returns the expected mapping using fake data.
    """
    monkeypatch.setattr("d2draftnet.collect_data.requests.get", fake_get_heroes)
    df_instance = DataFetcher("https://api.opendota.com/api")
    mapping = df_instance._openDotaID_2_heroname()
    assert mapping[1] == "Anti-Mage"
    assert mapping[2] == "Axe"

def test_is_duplicate_existing(data_fetcher):
    """
    Test that _is_duplicate returns True if the match_id exists in the stored dataset.
    """
    # match_id 1001 exists.
    assert data_fetcher._is_duplicate(1001, []) is True

def test_is_duplicate_new(data_fetcher):
    """
    Test that _is_duplicate returns False for a match_id not in either stored or new matches.
    """
    # match_id 2000 is not present.
    assert data_fetcher._is_duplicate(2000, []) is False

def test_is_duplicate_in_new_matches(data_fetcher):
    """
    Test that _is_duplicate returns True if the match_id is found in the new matches list.
    """
    new_matches = [
        {
            "match_id": 3001,
            "radiant_draft": ["Queen of Pain"],
            "dire_draft": ["Dragon Knight"],
            "duration": 3600,
            "winner": "Radiant",
        }
    ]
    assert data_fetcher._is_duplicate(3001, new_matches) is True
    assert data_fetcher._is_duplicate(4000, new_matches) is False

def test_analyze_pub_data(monkeypatch, tmp_path):
    """
    Test that analyze_pub_data collects the correct number of unique matches.
    We monkey-patch _get_pub_data to return fake public match data.
    """
    df_instance = DataFetcher("https://api.opendota.com/api")
    df_instance.hero_id_to_name = HERO_MAP
    df_instance.match_data = []  # Start with empty dataset.
    df_instance.output_path = tmp_path / "match_data.parquet"
    
    # Monkey-patch _get_pub_data to return our fake public matches.
    monkeypatch.setattr(df_instance, "_get_pub_data", fake_get_pub_data)
    # Collect 2 matches (should ignore the duplicate match_id 4001).
    df_instance.analyze_pub_data(N_matches=2)
    
    # Verify that 2 unique matches were collected.
    assert len(df_instance.match_data) == 2
    match_ids = [m["match_id"] for m in df_instance.match_data]
    assert len(match_ids) == len(set(match_ids))

