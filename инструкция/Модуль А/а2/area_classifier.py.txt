def classify_area(road_density: float, building_density: float, water_density: float) -> str:
    if water_density > 0.4:
        return "water"
    if building_density > 0.4:
        return "urban"
    if road_density < 0.2 and building_density < 0.1:
        return "forest"
    return "mixed"
