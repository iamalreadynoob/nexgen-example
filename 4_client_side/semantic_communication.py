"""# Semantic Communication.

It applies the semantic communication at the client side.
"""
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from nexgenpy.manager import Actuator

load_dotenv(
    override=True,
)

logger = logging.getLogger(__name__)

actuator = Actuator(
    name="gaming_habits",
    dst=os.getenv("NEXGEN_BASE_URL"),
    dst_headers={
        "X-API-KEY": os.getenv("NEXGEN_API_KEY"),
    },
    dst_save_original=True,
    try_load_actuator=True, # default: True
)

df = pd.read_csv(
    Path(__file__).parent.parent / "resources" / "client_side_example.csv",
)
df = df.rename(
    columns={
        "Age": "age",
        "Gender": "gender",
        "Location": "location",
        "GameGenre": "game_genre",
        "PlayTimeHours": "play_time_hours",
        "InGamePurchases": "in_game_purchases",
        "GameDifficulty": "game_difficulty",
        "SessionsPerWeek": "sessions_per_week",
        "AvgSessionDurationMinutes": "avg_session_duration_minutes",
        "PlayerLevel": "player_level",
        "AchievementsUnlocked": "achievements_unlocked",
    },
)
df = df.drop(
    columns=["PlayerID", "EngagementLevel"],
)

for sample in df.to_dict(orient="records"):
    actuator.run(sample)
