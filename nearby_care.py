"""
Nearby doctor / hospital lookup using free OpenStreetMap services:
- Nominatim for geocoding a typed place name into lat/lon
- Overpass API for finding hospitals, clinics, doctors, and pharmacies
  within a radius of a point

No API key required. Optional browser geolocation ("use my current
location") is used only if the `streamlit-js-eval` package is installed;
otherwise the app falls back to a typed place name, which always works.
"""

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "DiaLLM-BD-research-prototype/1.0"}

AMENITY_TYPES = ["hospital", "clinic", "doctors", "pharmacy"]


def geocode_place(place_name, country_bias="Bangladesh"):
    params = {
        "q": f"{place_name}, {country_bias}" if country_bias else place_name,
        "format": "json",
        "limit": 1,
    }
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=8)
        r.raise_for_status()
        results = r.json()
        if not results:
            return None
        return {"lat": float(results[0]["lat"]), "lon": float(results[0]["lon"]), "display_name": results[0]["display_name"]}
    except Exception:
        return None


def find_nearby_care(lat, lon, radius_km=5):
    radius_m = int(radius_km * 1000)
    amenity_filter = "|".join(AMENITY_TYPES)
    query = f"""
    [out:json][timeout:20];
    (
      node["amenity"~"^({amenity_filter})$"](around:{radius_m},{lat},{lon});
      way["amenity"~"^({amenity_filter})$"](around:{radius_m},{lat},{lon});
    );
    out center 30;
    """
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=20)
        r.raise_for_status()
        elements = r.json().get("elements", [])
    except Exception:
        return []

    results = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        el_lat = el.get("lat") or el.get("center", {}).get("lat")
        el_lon = el.get("lon") or el.get("center", {}).get("lon")
        if el_lat is None or el_lon is None:
            continue
        results.append({
            "name": name,
            "type": tags.get("amenity", "unknown"),
            "lat": el_lat,
            "lon": el_lon,
            "distance_km": _haversine_km(lat, lon, el_lat, el_lon),
            "phone": tags.get("phone") or tags.get("contact:phone"),
        })

    results.sort(key=lambda x: x["distance_km"])
    return results


def _haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return r * 2 * atan2(sqrt(a), sqrt(1 - a))
