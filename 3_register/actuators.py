"""# Actuators.

This file contains the encoder actuator definition for this example.
"""
import logging

from dotenv import load_dotenv
from nexgenpy.actuators import actuator

load_dotenv(
    override=True,
)

logger = logging.getLogger(__name__)

@actuator(
    name="gaming_habits",
    atype="coding",
    version="v1",
    is_latest=True,
    requirements=[
        "catboost==1.2.10",
        "boto3==1.43.117",
    ],
)
def classify_client(
    age: int,
    gender: str,
    location: str,
    game_genre: str,
    play_time_hours: float,
    in_game_purchases: int,
    game_difficulty: str,
    sessions_per_week: int,
    avg_session_duration_minutes: int,
    player_level: int,
    achievements_unlocked: int,
) -> tuple[int, float]:
    """# Classify Client.

    It predicts the engagement level of a player with the CatBoost model. The
    model and the mapping files are downloaded from the object storage into the
    actuator_resources folder if they are not there yet.

    Args:
        age (int): Age of the player.
        gender (str): Gender of the player.
        location (str): Location of the player.
        game_genre (str): Genre of the game.
        play_time_hours (float): Average play time in hours.
        in_game_purchases (int): 1 if the player makes in-game purchases, else 0.
        game_difficulty (str): Difficulty level of the game.
        sessions_per_week (int): Number of gaming sessions per week.
        avg_session_duration_minutes (int): Average session duration in minutes.
        player_level (int): Current level of the player.
        achievements_unlocked (int): Number of unlocked achievements.

    Returns:
        tuple[int, float]: The predicted class number and its probability.
    """
    import json
    import os

    import boto3
    from catboost import CatBoostClassifier

    s3_client = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
        region_name=os.getenv("MINIO_REGION"),
    )

    bucket_name = os.getenv("MINIO_BUCKET")
    model_key = os.getenv("MINIO_MODEL_KEY")

    # Downloading The Missing Objects
    resources_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "actuator_resources",
    )
    os.makedirs(resources_dir, exist_ok=True)

    object_keys = [
        model_key,
        "gender_map.json",
        "location_map.json",
        "game_genre_map.json",
        "game_difficulty_map.json",
    ]

    for key in object_keys:
        file_path = os.path.join(resources_dir, key)
        if os.path.exists(file_path) is False:
            s3_client.download_file(bucket_name, key, file_path)

    def load_map(key: str) -> dict[str, int]:
        with open(os.path.join(resources_dir, key)) as f:
            return json.load(f)

    gender_map = load_map("gender_map.json")
    location_map = load_map("location_map.json")
    game_genre_map = load_map("game_genre_map.json")
    game_difficulty_map = load_map("game_difficulty_map.json")

    # Loading The Model
    model = CatBoostClassifier()
    model.load_model(os.path.join(resources_dir, model_key))

    # Preparing The Input, same column order as in train.ipynb
    features = [
        age,
        gender_map[gender],
        location_map[location],
        game_genre_map[game_genre],
        play_time_hours,
        in_game_purchases,
        game_difficulty_map[game_difficulty],
        sessions_per_week,
        avg_session_duration_minutes,
        player_level,
        achievements_unlocked,
    ]

    # Prediction
    probabilities = model.predict_proba([features])[0]
    predicted_class = int(probabilities.argmax())

    return predicted_class, float(probabilities[predicted_class])

@actuator(
    name="gaming_habits_decode",
    atype="decoding",
    version="v2",
    is_latest=True,
    requirements=[
        "boto3==1.43.117",
        "requests==2.34.2",
    ],
)
def print_out_response() -> None:
    """# Print Out Response.

    It prints out the response class name with its probability.
    """
    import json
    import os

    import boto3
    import requests

    s3_client = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
        region_name=os.getenv("MINIO_REGION"),
    )

    bucket_name = os.getenv("MINIO_BUCKET")

    # Downloading The Missing Objects
    resources_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "actuator_resources",
    )
    os.makedirs(resources_dir, exist_ok=True)

    file_path = os.path.join(resources_dir, "engagement_level_map.json")
    if os.path.exists(file_path) is False:
        s3_client.download_file(bucket_name, "engagement_level_map.json", file_path)

    def load_map(key: str) -> dict[str, int]:
        with open(os.path.join(resources_dir, key)) as f:
            return json.load(f)

    engagement_level_map = load_map("engagement_level_map.json")
    engagement_level_map = {v: k for k, v in engagement_level_map.items()}

    # Fetching The Latest Encoder Actuator ID
    response = requests.get(
        f"{os.getenv("NEXGEN_BASE_URL")}/actuators/gaming_habits/latest",
        headers={
            "X-API-KEY": os.getenv("NEXGEN_API_KEY"),
        },
        timeout=15,
    )
    response.raise_for_status()

    actuator_id = response.json()["_id"]

    response = requests.get(
        f"{os.getenv("NEXGEN_BASE_URL")}/data",
        headers={
            "X-API-KEY": os.getenv("NEXGEN_API_KEY"),
        },
        params={
            "page": 1,
            "limit": 10,
            "actuator_id": actuator_id,
        },
        timeout=15,
    )
    response.raise_for_status()

    # Printing The Responses, metadata is [predicted_class, probability]
    for item in response.json()["data"]:
        predicted_class, probability = item["metadata"]
        print(
            f"{item['created_at']} | {item['linked_resource']} | "
            f"{engagement_level_map[predicted_class]} ({probability:.2%})",
        )


if __name__ == "__main__":
    # Logging Config, it also shows the request logs of nexgenpy
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # The decorator publishes the actuator instead of running it,
    # so the arguments are not needed
    for register in (classify_client, print_out_response):
        is_registered = register()

        if is_registered is True:
            logger.info("Actuator registered: %s", is_registered)
        else:
            logger.error("Actuator registered: %s", is_registered)
