
import requests

url = "https://api.dharun.dev/v1/mandi/commodities"

params = {
    "state": "Tamil Nadu"
}

try:
    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    print("API Connected Successfully!")

    print("Latest Tamil Nadu commodities:")

    for item in result.get("data", []):
        print(item)

except requests.exceptions.RequestException as error:
    print("Connection Error:", error)
