"""
Helper utility functions for the application
Contains reusable routines such as response formatting, file validation, etc.
"""


def format_api_response(status: str, message: str, data: dict = None):
    """
    Standardize JSON response structure across the API.
    """
    response = {
        "status": status,
        "message": message
    }
    if data is not None:
        response["data"] = data
    return response
