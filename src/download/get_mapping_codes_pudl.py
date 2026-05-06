import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

url = "https://docs.catalyst.coop/pudl/en/latest/data_dictionaries/codes_and_labels.html"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")


def get_pudl_code_table(anchor_id):
    section = soup.find(id=anchor_id)

    if section is None:
        raise ValueError(f"Could not find section: {anchor_id}")

    table_html = section.find_next("table")

    if table_html is None:
        raise ValueError(f"Could not find table after section: {anchor_id}")

    return pd.read_html(StringIO(str(table_html)))[0]


energy_sources = get_pudl_code_table("core-eia-codes-energy-sources")
prime_movers = get_pudl_code_table("core-eia-codes-prime-movers")

energy_sources.to_csv("../../../input_data/core_eia__codes_energy_sources.csv", index=False)
prime_movers.to_csv("../../../input_data/core_eia__codes_prime_movers.csv", index=False)