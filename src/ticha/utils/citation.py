"""Citation and metadata utilities."""

from datetime import datetime
from typing import Dict


def generate_citation_info(url: str) -> Dict[str, str]:
    """Generate proper citation information for scraped data."""
    return {
        'url': url,
        'access_date': datetime.now().isoformat(),
        'source': 'Ticha: a digital text explorer for Colonial Zapotec',
        'citation': f"Data retrieved from {url}, accessed {datetime.now().strftime('%Y-%m-%d')}",
        'project_info': 'Ticha project by Brook Danielle Lillehaugen, George Aaron Broadwell, et al.'
    }
