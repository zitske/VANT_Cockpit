import math


sim_data = {
    "pitch": 0.0,
    "roll": 0.0,
    "heading": 0.0,
    "airspeed": 0.0,
    "altitude": 0.0,
    "ground_speed": 0.0,
    "lat": -22.9068,
    "lon": -43.1729,
    "home_lat": -22.9068,
    "home_lon": -43.1729,
    "sats": 10,
    "batt_volt": 16.8,
    "flight_mode": "STABILIZE",
    "thermal_is_main": True,
}


def update_simulation(state, current_time, delta_time):
    t = current_time * 0.5
    state["roll"] = math.sin(t * 0.7) * 30
    state["pitch"] = math.cos(t * 0.5) * 15
    state["heading"] = (state["heading"] + delta_time * 5) % 360
    state["altitude"] = 100 + (math.sin(t * 0.2) * 20)
    state["airspeed"] = 20 + (math.sin(t * 0.3) * 5)
    state["ground_speed"] = state["airspeed"] - 1.5
    state["sats"] = 12 + int(math.sin(t))
    state["batt_volt"] -= delta_time * 0.01
    state["lon"] += delta_time * 0.0001
