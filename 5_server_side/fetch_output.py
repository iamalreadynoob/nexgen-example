"""# Fetch Output.

It prints out the sent output.
"""
import logging
import os

from dotenv import load_dotenv
from nexgenpy.manager import Actuator

load_dotenv(
    override=True,
)

logger = logging.getLogger(__name__)

actuator = Actuator(
    name="gaming_habits_decode",
    dst=os.getenv("NEXGEN_BASE_URL"),
    dst_headers={
        "X-API-KEY": os.getenv("NEXGEN_API_KEY"),
    },
    dst_save_original=True,
    try_load_actuator=True, # default: True
)

actuator.run(
    data={},
)
