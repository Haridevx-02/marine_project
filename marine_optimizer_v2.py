import streamlit as st
import requests
from geopy.geocoders import Nominatim
from datetime import datetime
import numpy as np
import logging
import random
import time
from functools import lru_cache
from groq import Groq
import threading
from math import radians, sin, cos, sqrt, atan2
import folium
from streamlit_folium import folium_static
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import searoute as sr
    _SEAROUTE_AVAILABLE = True
except ImportError:
    _SEAROUTE_AVAILABLE = False


def get_sea_route_coords(start_coords: Tuple[float, float],
                         end_coords: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Return a list of (lat, lon) tuples following real maritime sea lanes.
    Falls back to a straight line if searoute is unavailable or fails."""
    if _SEAROUTE_AVAILABLE:
        try:
            # searoute expects [lon, lat] order
            route = sr.searoute(
                [start_coords[1], start_coords[0]],
                [end_coords[1],   end_coords[0]]
            )
            # GeoJSON coordinates are [lon, lat] — flip to (lat, lon) for folium
            coords = [(pt[1], pt[0]) for pt in route['geometry']['coordinates']]
            if len(coords) >= 2:
                return coords
        except Exception:
            pass
    # Fallback: straight great-circle line with interpolated points
    pts = []
    for i in range(11):
        r = i / 10
        pts.append((
            start_coords[0] + r * (end_coords[0] - start_coords[0]),
            start_coords[1] + r * (end_coords[1] - start_coords[1]),
        ))
    return pts

# Configure page
st.set_page_config(
    page_title="Marine Route Optimizer V2",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Replace these with your actual API keys via st.secrets
WEATHER_API_KEY = st.secrets.get("WEATHER_API_KEY", "YOUR_WEATHER_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

# Configure logging
logging.basicConfig(filename='marine_optimization.log', level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class WeatherData:
    location: str
    temperature: float
    feels_like: float
    description: str
    wind_speed: float
    wind_direction: float
    humidity: int
    pressure: float
    visibility: float
    timestamp: datetime

@dataclass
class Ship:
    ship_type: str
    max_speed: float
    fuel_consumption: float
    safety_rating: float
    fuel_capacity: float  # in liters or tons
    vessel_weight: float  # in tons

class MarineWeatherAnalyzer:
    def __init__(self):
        """Initialize the analyzer with API clients and caching"""
        self.weather_api_key = WEATHER_API_KEY
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.geolocator = Nominatim(
            user_agent="marine_optimization_v2.0",
            timeout=5
        )
        
    @lru_cache(maxsize=100)
    def get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates with caching for faster repeated lookups"""
        location = location.lower().strip()

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logging.info(f"Attempting to geocode location: {location} (attempt {attempt + 1})")
                    loc = self.geolocator.geocode(location)
                    if loc:
                        coords = (loc.latitude, loc.longitude)
                        logging.info(f"Successfully geocoded location: {location}")
                        return coords
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to geocode after {max_retries} attempts: {e}")
                        raise
                    logging.warning(f"Geocoding attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
            logging.warning(f"Could not find coordinates for location: {location}")
            return None
        except Exception as e:
            logging.error(f"Error fetching coordinates: {e}")
            return None

    def fetch_weather_data(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Fetch and parse marine weather data with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = (
                    f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
                )
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                location = data.get('name', 'Open Water')
                if not location or location == '':
                    location = f"Coordinates: {lat:.2f}°N, {lon:.2f}°E"
                
                weather_data = WeatherData(
                    location=location,
                    temperature=data['main']['temp'],
                    feels_like=data['main']['feels_like'],
                    description=data['weather'][0]['description'],
                    wind_speed=data['wind'].get('speed', 0),
                    wind_direction=data['wind'].get('deg', 0),
                    humidity=data['main']['humidity'],
                    pressure=data['main']['pressure'],
                    visibility=data.get('visibility', 0) / 1000,  # Convert to km
                    timestamp=datetime.fromtimestamp(data['dt'])
                )
                logging.info(f"Successfully fetched weather data for {location}")
                return weather_data

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to fetch weather data after {max_retries} attempts: {e}")
                    return None
                logging.warning(f"Weather data fetch attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            except KeyError as e:
                logging.error(f"Error parsing weather data: {e}")
                return None

    def ant_colony_optimization(self, graph, start, end, ship: Ship, weather_data: WeatherData, num_ants=10, max_iterations=100, alpha=1, beta=2, evaporation_rate=0.5):
        """Implement Ant Colony Optimization with fuel, weight, and weather considerations"""
        pheromones = np.ones_like(graph, dtype=float)
        best_route = None
        best_cost = float('inf')
        
        convergence_count = 0
        prev_best_cost = float('inf')
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for iteration in range(max_iterations):
            routes = []
            for ant in range(num_ants):
                route = self.construct_route(graph, start, end, pheromones, alpha, beta, ship, weather_data)
                cost = self.calculate_cost(route, graph, ship, weather_data)
                routes.append((route, cost))

                if cost < best_cost:
                    best_route = route
                    best_cost = cost
                    convergence_count = 0
                elif cost == prev_best_cost:
                    convergence_count += 1
                    
                if convergence_count >= 10:
                    logging.info(f"ACO converged after {iteration + 1} iterations")
                    status_text.text(f"Algorithm converged after {iteration + 1} iterations")
                    progress_bar.progress(1.0)
                    return best_route, best_cost

            prev_best_cost = best_cost
            self.update_pheromones(pheromones, routes, evaporation_rate)
            
            # Update progress bar
            progress_bar.progress((iteration + 1) / max_iterations)
            status_text.text(f"Iteration {iteration + 1}/{max_iterations}: Best cost = {best_cost:.2f}")

        return best_route, best_cost

    def construct_route(self, graph, start, end, pheromones, alpha, beta, ship: Ship, weather_data: WeatherData):
        """Construct a route for an ant considering fuel, weight, and weather"""
        route = [start]
        current = start
        visited = {start}

        while current != end:
            neighbors = [n for n in range(len(graph)) if graph[current][n] != 0 and n not in visited]
            if not neighbors:
                break
                
            probabilities = self.calculate_probabilities(current, neighbors, pheromones, graph, alpha, beta, ship, weather_data)
            next_node = random.choices(neighbors, probabilities)[0]
            route.append(next_node)
            visited.add(next_node)
            current = next_node

        return route

    def calculate_probabilities(self, current, neighbors, pheromones, graph, alpha, beta, ship: Ship, weather_data: WeatherData):
        """Calculate transition probabilities considering fuel, weight, and weather"""
        probabilities = []
        total = 0

        for neighbor in neighbors:
            pheromone = pheromones[current][neighbor] ** alpha
            heuristic = (1 / self.calculate_edge_cost(current, neighbor, graph, ship, weather_data)) ** beta
            probability = pheromone * heuristic
            probabilities.append(probability)
            total += probability

        return [p / total if total > 0 else 1 / len(neighbors) for p in probabilities]

    def calculate_edge_cost(self, current, neighbor, graph, ship: Ship, weather_data: WeatherData):
        """Calculate the cost of an edge considering fuel, weight, and weather"""
        base_cost = graph[current][neighbor]
        
        # Adjust cost based on weather conditions
        weather_factor = 1 + (weather_data.wind_speed / 10)  # Higher wind speed increases cost
        if weather_data.visibility < 5:  # Low visibility increases cost
            weather_factor *= 1.5
        
        # Adjust cost based on fuel consumption and vessel weight
        fuel_factor = (ship.fuel_consumption * base_cost) / ship.fuel_capacity
        weight_factor = ship.vessel_weight / 1000  # Heavier vessels have higher costs
        
        return base_cost * weather_factor * (1 + fuel_factor + weight_factor)

    def calculate_cost(self, route, graph, ship: Ship, weather_data: WeatherData):
        """Calculate route cost considering fuel, weight, and weather"""
        return sum(self.calculate_edge_cost(route[i], route[i + 1], graph, ship, weather_data) for i in range(len(route) - 1))

    def update_pheromones(self, pheromones, routes, evaporation_rate):
        """Update pheromone levels"""
        pheromones *= (1 - evaporation_rate)
        for route, cost in routes:
            deposit = 1.0 / cost if cost > 0 else 1.0
            for i in range(len(route) - 1):
                pheromones[route[i]][route[i + 1]] += deposit

    def generate_route_summary(self, route, weather_data: WeatherData, ship: Ship) -> str:
        """Generate a concise route summary with key details"""
        try:
            optimized_distance = len(route) - 1  # Assuming distance is based on route length
            summary = f"""
            ## Optimized Route Summary
            - Start: {route[0]}
            - End: {route[-1]}
            - Total Distance: {optimized_distance} nautical miles
            
            ## Vessel Details
            - Ship Type: {ship.ship_type}
            - Max Speed: {ship.max_speed} knots
            - Fuel Capacity: {ship.fuel_capacity} liters/tons
            - Vessel Weight: {ship.vessel_weight} tons
            
            ## Recommendations
            - Maintain optimal speed of {ship.max_speed * 0.8:.1f} knots for fuel efficiency.
            - Monitor fuel consumption closely.
            - Be prepared for potential weather changes.
            """
            return summary.strip()
        except Exception as e:
            logging.error(f"Error generating route summary: {e}")
            return "Error generating route summary. Please check the logs for details."

    def haversine_distance(self, coord1, coord2):
        """Calculate the great-circle distance between two points on the Earth."""
        R = 6371.0  # Radius of the Earth in kilometers
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c  # Distance in kilometers

    def calculate_eta(self, start_coords, end_coords, ship: Ship) -> float:
        """Calculate the estimated time of arrival based on the start and end coordinates and ship conditions."""
        distance_km = self.haversine_distance(start_coords, end_coords)  # Get distance in kilometers
        
        # Convert distance to nautical miles (1 km = 0.539957 nautical miles)
        distance_nautical_miles = distance_km * 0.539957
        
        # Calculate speed adjustment based on wind speed
        wind_adjustment = 1 + (ship.max_speed * 0.1 * (ship.fuel_consumption / ship.fuel_capacity))  # Example adjustment
        effective_speed = ship.max_speed / wind_adjustment  # Adjusted speed considering wind and fuel

        # Calculate time in hours
        time_hours = distance_nautical_miles / effective_speed if effective_speed > 0 else float('inf')
        return time_hours

    def calculate_fuel_consumption(self, distance_nautical_miles: float, ship: Ship) -> float:
        """Calculate the fuel consumed based on distance and ship specifications."""
        return distance_nautical_miles * ship.fuel_consumption

    def update_fuel_capacity(self, initial_fuel_capacity: float, distance_nautical_miles: float, ship: Ship) -> float:
        """Update the fuel capacity based on distance traveled."""
        fuel_consumed = self.calculate_fuel_consumption(distance_nautical_miles, ship)
        return max(0, initial_fuel_capacity - fuel_consumed)  # Ensure fuel doesn't go below 0

    def create_map(self, start_coords, end_coords, waypoints=None):
        """Create a map with the optimized sea route following real maritime lanes."""
        # Center the map between start and end
        center = (
            (start_coords[0] + end_coords[0]) / 2,
            (start_coords[1] + end_coords[1]) / 2,
        )
        m = folium.Map(location=center, zoom_start=4, tiles="CartoDB positron")

        # Add OpenSeaMap overlay for maritime chart detail
        folium.TileLayer(
            tiles='https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png',
            attr='OpenSeaMap',
            name='Sea Marks',
            overlay=True,
            control=True,
            opacity=0.7
        ).add_to(m)

        # Compute the real sea route (avoids land)
        sea_route = get_sea_route_coords(start_coords, end_coords)

        # Draw the sea-lane-following polyline
        folium.PolyLine(
            locations=sea_route,
            color='#0057b8',
            weight=4,
            opacity=0.85,
            tooltip='Optimized Sea Route'
        ).add_to(m)

        # Start / end markers
        folium.Marker(
            start_coords, tooltip='Start Location',
            icon=folium.Icon(color='green', icon='ship', prefix='fa')
        ).add_to(m)
        folium.Marker(
            end_coords, tooltip='End Location',
            icon=folium.Icon(color='red', icon='flag', prefix='fa')
        ).add_to(m)

        # Mark a few intermediate sea-route waypoints
        step = max(1, len(sea_route) // 5)
        for i, wp in enumerate(sea_route[step:-1:step], start=1):
            folium.CircleMarker(
                location=wp,
                radius=4,
                color='#0057b8',
                fill=True,
                fill_opacity=0.7,
                tooltip=f'Waypoint {i}'
            ).add_to(m)

        folium.LayerControl().add_to(m)
        return m

def create_fuel_consumption_chart(distance, ship):
    """Create a chart showing estimated fuel consumption over distance"""
    distances = np.linspace(0, distance * 1.5, 20)  # Create distance points with margin
    consumptions = [ship.fuel_consumption * d for d in distances]
    
    # Create DataFrame for the chart
    df = pd.DataFrame({
        'Distance (nautical miles)': distances,
        'Fuel Consumption (liters/tons)': consumptions
    })
    
    # Highlight when fuel consumption exceeds capacity
    threshold_distance = ship.fuel_capacity / ship.fuel_consumption
    
    fig = px.line(df, x='Distance (nautical miles)', y='Fuel Consumption (liters/tons)', 
                  title='Estimated Fuel Consumption Over Distance')
    
    # Add threshold line for fuel capacity
    fig.add_hline(y=ship.fuel_capacity, line_dash="dash", line_color="red",
                 annotation_text=f"Fuel Capacity ({ship.fuel_capacity} liters/tons)",
                 annotation_position="top right")
    
    # Add marker for journey distance
    fig.add_vline(x=distance, line_dash="dash", line_color="green",
                 annotation_text=f"Journey Distance ({distance:.1f} nautical miles)",
                 annotation_position="top left")
    
    return fig

def create_weather_radar_chart(weather_data):
    """Create a radar chart to visualize weather conditions"""
    # Normalize the values for radar chart (0-1 scale)
    temperature_norm = min(1.0, weather_data.temperature / 40)  # Normalize temperature (assuming max 40°C)
    wind_speed_norm = min(1.0, weather_data.wind_speed / 20)    # Normalize wind speed (assuming max 20 m/s)
    humidity_norm = weather_data.humidity / 100                  # Already in percentage
    pressure_norm = (weather_data.pressure - 950) / 150          # Normalize pressure (assuming range 950-1100 hPa)
    visibility_norm = min(1.0, weather_data.visibility / 10)     # Normalize visibility (assuming max 10 km)
    
    categories = ['Temperature', 'Wind Speed', 'Humidity', 'Pressure', 'Visibility']
    values = [temperature_norm, wind_speed_norm, humidity_norm, pressure_norm, visibility_norm]
    
    # Close the polygon by repeating the first value
    values = values + [values[0]]
    categories = categories + [categories[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Weather Conditions'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Weather Condition Analysis"
    )
    
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# ML HAZARD PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
class HazardPredictor:
    """Machine-learning model that predicts hazard level and voyage delay."""

    HAZARD_LABELS = ["Low", "Medium", "High"]
    FEATURES = ["wind_speed", "visibility", "temperature", "humidity", "pressure"]

    def __init__(self):
        self._clf: Optional[Pipeline] = None
        self._reg: Optional[Pipeline] = None
        self._train()

    # ------------------------------------------------------------------
    # Training on synthetic data (representative of real-world patterns)
    # ------------------------------------------------------------------
    def _generate_training_data(self, n: int = 2000):
        rng = np.random.default_rng(42)
        wind_speed   = rng.uniform(0, 30, n)          # m/s
        visibility   = rng.uniform(0.5, 20, n)        # km
        temperature  = rng.uniform(-5, 42, n)         # °C
        humidity     = rng.uniform(20, 100, n)        # %
        pressure     = rng.uniform(960, 1050, n)      # hPa

        # Rule-based hazard labelling
        risk = np.zeros(n)
        risk += (wind_speed > 15) * 2
        risk += (wind_speed > 8)  * 1
        risk += (visibility < 3)  * 2
        risk += (visibility < 7)  * 1
        risk += (pressure < 985)  * 1
        risk += (np.abs(temperature) > 35) * 1
        hazard = np.where(risk >= 4, 2, np.where(risk >= 2, 1, 0))  # 0=Low,1=Med,2=High

        # Delay in hours (proportional to risk score + noise)
        delay = risk * 1.5 + rng.uniform(0, 2, n)

        X = pd.DataFrame({
            "wind_speed": wind_speed,
            "visibility": visibility,
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
        })
        return X, hazard, delay

    def _train(self):
        X, y_hazard, y_delay = self._generate_training_data()
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self._reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        self._clf.fit(X, y_hazard)
        self._reg.fit(X, y_delay)

    def predict(self, weather: "WeatherData") -> Dict:
        """Return hazard label, confidence, delay estimate, and feature importances."""
        X = pd.DataFrame([{
            "wind_speed":  weather.wind_speed,
            "visibility":  weather.visibility,
            "temperature": weather.temperature,
            "humidity":    weather.humidity,
            "pressure":    weather.pressure,
        }])
        hazard_idx   = int(self._clf.predict(X)[0])
        confidences  = self._clf.predict_proba(X)[0] * 100
        delay_hrs    = float(self._reg.predict(X)[0])
        importances  = self._clf.named_steps["clf"].feature_importances_
        return {
            "hazard_label":    self.HAZARD_LABELS[hazard_idx],
            "hazard_idx":      hazard_idx,
            "confidences":     dict(zip(self.HAZARD_LABELS, confidences)),
            "delay_hours":     max(0.0, delay_hrs),
            "feature_importance": dict(zip(self.FEATURES, importances)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-STOP VOYAGE OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class MultiStopOptimizer:
    """Optimises a multi-port itinerary using greedy nearest-neighbour ordering."""

    def __init__(self, analyzer: "MarineWeatherAnalyzer"):
        self.analyzer = analyzer

    def _nn_order(self, ports: List[str], coords: Dict[str, Tuple]) -> List[str]:
        """Greedy nearest-neighbour to order intermediate ports (origin & dest fixed)."""
        origin = ports[0]
        destination = ports[-1]
        intermediates = list(ports[1:-1])
        ordered = [origin]
        current_coords = coords[origin]
        remaining = intermediates[:]
        while remaining:
            nearest = min(
                remaining,
                key=lambda p: self.analyzer.haversine_distance(current_coords, coords[p])
            )
            ordered.append(nearest)
            current_coords = coords[nearest]
            remaining.remove(nearest)
        ordered.append(destination)
        return ordered

    def optimize(self, ports: List[str], ship: "Ship") -> Dict:
        """Resolve coordinates, order stops, compute per-leg metrics."""
        coords: Dict[str, Optional[Tuple]] = {}
        for port in ports:
            c = self.analyzer.get_coordinates(port)
            if c is None:
                return {"error": f"Could not geocode port: {port}"}
            coords[port] = c

        ordered = self._nn_order(ports, coords)

        legs = []
        total_dist = total_eta = total_fuel = 0.0
        for i in range(len(ordered) - 1):
            src, dst = ordered[i], ordered[i + 1]
            dist_nm = self.analyzer.haversine_distance(coords[src], coords[dst]) * 0.539957
            eta_h   = self.analyzer.calculate_eta(coords[src], coords[dst], ship)
            fuel    = self.analyzer.calculate_fuel_consumption(dist_nm, ship)
            legs.append({
                "From": src, "To": dst,
                "Distance (nm)": round(dist_nm, 1),
                "ETA (hrs)": round(eta_h, 2),
                "Fuel Est.": round(fuel, 1),
            })
            total_dist += dist_nm
            total_eta  += eta_h
            total_fuel += fuel

        return {
            "ordered_ports": ordered,
            "coords": coords,
            "legs": legs,
            "total_distance": round(total_dist, 1),
            "total_eta": round(total_eta, 2),
            "total_fuel": round(total_fuel, 1),
        }

    def build_map(self, result: Dict) -> folium.Map:
        ordered = result["ordered_ports"]
        coords  = result["coords"]
        center  = coords[ordered[len(ordered) // 2]]
        m = folium.Map(location=center, zoom_start=3, tiles="CartoDB positron")

        # OpenSeaMap overlay
        folium.TileLayer(
            tiles='https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png',
            attr='OpenSeaMap', name='Sea Marks',
            overlay=True, control=True, opacity=0.7
        ).add_to(m)

        # Draw real sea route for each leg
        leg_colors = ['#0057b8', '#0096c7', '#00b4d8', '#48cae4', '#90e0ef']
        for i in range(len(ordered) - 1):
            src, dst = ordered[i], ordered[i + 1]
            sea_leg = get_sea_route_coords(coords[src], coords[dst])
            folium.PolyLine(
                locations=sea_leg,
                color=leg_colors[i % len(leg_colors)],
                weight=4, opacity=0.85,
                tooltip=f'Leg {i+1}: {src} → {dst}'
            ).add_to(m)

        # Port markers
        port_colors = ["green"] + ["orange"] * (len(ordered) - 2) + ["red"]
        for i, port in enumerate(ordered):
            folium.Marker(
                coords[port],
                tooltip=f"{i+1}. {port}",
                icon=folium.Icon(color=port_colors[i], icon='ship', prefix='fa')
            ).add_to(m)

        folium.LayerControl().add_to(m)
        return m


def main():
    st.title("🚢 Marine Route Optimizer V2")
    st.markdown("""
    This application helps optimize maritime routes based on vessel specifications, 
    weather conditions, and ant colony optimization algorithms.
    """)
    
    # Initialize session state for persistence
    if 'optimized' not in st.session_state:
        st.session_state.optimized = False
    if 'multi_optimized' not in st.session_state:
        st.session_state.multi_optimized = False
    
    # Sidebar for inputs
    st.sidebar.header("Navigation Parameters")
    
    # Locations
    start_location = st.sidebar.text_input("Start Location", "San Francisco")
    end_location = st.sidebar.text_input("End Location", "Los Angeles")
    
    # Vessel details
    st.sidebar.header("Vessel Specifications")
    
    ship_type = st.sidebar.selectbox(
        "Ship Type",
        ["Cargo", "Tanker", "Passenger", "Container", "Fishing"]
    )
    
    max_speed = st.sidebar.slider("Maximum Speed (knots)", 5, 40, 20)
    fuel_capacity = st.sidebar.number_input("Fuel Capacity (liters/tons)", min_value=100, max_value=10000, value=1000)
    vessel_weight = st.sidebar.number_input("Vessel Weight (tons)", min_value=10, max_value=5000, value=50)
    
    # Advanced settings
    st.sidebar.header("Advanced Settings")
    show_advanced = st.sidebar.checkbox("Show Advanced Settings")
    
    if show_advanced:
        fuel_consumption = st.sidebar.slider("Fuel Consumption Rate", 0.01, 1.0, 0.1, 0.01)
        safety_rating = st.sidebar.slider("Safety Rating", 0.1, 1.0, 0.9, 0.1)
        num_ants = st.sidebar.slider("Number of Ants (Algorithm Parameter)", 5, 50, 10)
        max_iterations = st.sidebar.slider("Maximum Iterations", 10, 500, 100)
    else:
        fuel_consumption = 0.1
        safety_rating = 0.9
        num_ants = 10
        max_iterations = 100
        
    # Create ship object
    ship = Ship(
        ship_type=ship_type,
        max_speed=max_speed,
        fuel_consumption=fuel_consumption,
        safety_rating=safety_rating,
        fuel_capacity=fuel_capacity,
        vessel_weight=vessel_weight
    )
    
    # Start button
    optimize_button = st.sidebar.button("Optimize Route", type="primary")
    
    # Initialize analyzer
    analyzer = MarineWeatherAnalyzer()
    
    # If button clicked, store in session state
    if optimize_button:
        with st.spinner("Optimizing your maritime route..."):
            try:
                # Fetch coordinates
                start_coords = analyzer.get_coordinates(start_location)
                end_coords = analyzer.get_coordinates(end_location)
                
                if not start_coords or not end_coords:
                    st.error("Error: Could not find coordinates for the specified locations. Please check the location names.")
                else:
                    # Store results in session state
                    st.session_state.start_coords = start_coords
                    st.session_state.end_coords = end_coords
                    st.session_state.weather_data = analyzer.fetch_weather_data(start_coords[0], start_coords[1])
                    
                    if not st.session_state.weather_data:
                        st.error("Error: Could not fetch weather data.")
                    else:
                        # Success
                        st.session_state.optimized = True
                        
                        # Graph and ACO
                        graph = np.array([[0, 1, 2, 0], [1, 0, 3, 4], [2, 3, 0, 5], [0, 4, 5, 0]])
                        optimized_route, cost = analyzer.ant_colony_optimization(
                            graph, start=0, end=3, ship=ship, weather_data=st.session_state.weather_data,
                            num_ants=num_ants, max_iterations=max_iterations
                        )
                        st.session_state.optimized_route = optimized_route
                        st.session_state.distance = analyzer.haversine_distance(start_coords, end_coords) * 0.539957
                        st.session_state.eta_hours = analyzer.calculate_eta(start_coords, end_coords, ship)
                        st.session_state.fuel_consumption_estimate = analyzer.calculate_fuel_consumption(st.session_state.distance, ship)
                        st.session_state.remaining_fuel = ship.fuel_capacity - st.session_state.fuel_consumption_estimate
            except Exception as e:
                st.error(f"Optimization error: {e}")

    # Display results if optimized
    if st.session_state.optimized:
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ Route Map", "🌤️ Weather Data",
            "📋 Route Analysis", "⛽ Fuel Analysis",
            "🤖 ML Hazard Prediction", "🚢 Multi-Stop Voyage"
        ])
        
        weather_data = st.session_state.weather_data
        
        # Tab 1: Map
        with tab1:
            st.header("Maritime Route Map")
            m = analyzer.create_map(st.session_state.start_coords, st.session_state.end_coords)
            folium_static(m)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Distance", f"{st.session_state.distance:.1f} nautical miles")
            col2.metric("ETA", f"{st.session_state.eta_hours:.1f} hours")
            col3.metric("Fuel Used", f"{st.session_state.fuel_consumption_estimate:.1f} liters/tons")
        
        # Tab 2: Weather data
        with tab2:
            st.header(f"Weather Conditions at {weather_data.location}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Temperature", f"{weather_data.temperature:.1f}°C")
            col2.metric("Wind Speed", f"{weather_data.wind_speed:.1f} m/s")
            col3.metric("Visibility", f"{weather_data.visibility:.1f} km")
            
            col1, col2 = st.columns(2)
            col1.metric("Humidity", f"{weather_data.humidity}%")
            col2.metric("Pressure", f"{weather_data.pressure} hPa")
            
            st.markdown(f"**Weather description:** {weather_data.description}")
            st.markdown(f"**Last updated:** {weather_data.timestamp}")
            st.plotly_chart(create_weather_radar_chart(weather_data))
            
            if weather_data.wind_speed > 10:
                st.warning("⚠️ High wind speeds detected!")
            if weather_data.visibility < 5:
                st.warning("⚠️ Low visibility conditions!")
        
        # Tab 3: Route analysis
        with tab3:
            st.header("Route Analysis and Recommendations")
            route_summary = analyzer.generate_route_summary(st.session_state.optimized_route, weather_data, ship)
            st.markdown(route_summary)
            
            danger_score = min(100, max(0, (weather_data.wind_speed / 15) * 100))
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = danger_score,
                gauge = {'axis': {'range': [None, 100]},
                         'bar': {'color': "darkblue"},
                         'steps' : [{'range': [0, 33], 'color': "green"},
                                    {'range': [33, 66], 'color': "yellow"},
                                    {'range': [66, 100], 'color': "red"}]}))
            st.plotly_chart(fig)
        
        # Tab 4: Fuel analysis
        with tab4:
            st.header("Fuel Consumption Analysis")
            col1, col2 = st.columns(2)
            col1.metric("Fuel Consumption", f"{st.session_state.fuel_consumption_estimate:.1f} L/T")
            col2.metric("Remaining Fuel", f"{st.session_state.remaining_fuel:.1f} L/T")
            
            st.plotly_chart(create_fuel_consumption_chart(st.session_state.distance, ship))
        
        # Tab 5: ML Hazard
        with tab5:
            st.header("🤖 ML Hazard & Delay Prediction")
            predictor = HazardPredictor()
            pred = predictor.predict(weather_data)
            st.subheader(f"Hazard level: **{pred['hazard_label']}**")
            st.info(f"⏱️ **Estimated Voyage Delay:** {pred['delay_hours']:.1f} hours")
            
            fi_df = pd.DataFrame(list(pred["feature_importance"].items()), columns=["Feature", "Importance"])
            st.plotly_chart(px.bar(fi_df, x="Importance", y="Feature", orientation="h"))
        
        # Tab 6: Multi-Stop (The fix is here - use session state to persist results)
        with tab6:
            st.header("🚢 Multi-Stop Voyage Optimizer")
            ports_input = st.text_area(
                "Ports (one per line — first = origin, last = destination):",
                value=f"{start_location}\n{end_location}", height=150
            )
            multi_btn = st.button("⚙️ Optimize Multi-Stop Route", key="ms_btn")
            
            if multi_btn:
                port_list = [p.strip() for p in ports_input.splitlines() if p.strip()]
                if len(port_list) >= 2:
                    with st.spinner("Optimizing multi-stop route..."):
                        ms_optimizer = MultiStopOptimizer(analyzer)
                        st.session_state.ms_result = ms_optimizer.optimize(port_list, ship)
                        st.session_state.multi_optimized = True
            
            if st.session_state.get('multi_optimized'):
                res = st.session_state.ms_result
                if "error" in res:
                    st.error(res["error"])
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Distance", f"{res['total_distance']} nm")
                    m2.metric("Total ETA", f"{res['total_eta']} hrs")
                    m3.metric("Total Fuel Est.", f"{res['total_fuel']} L/T")
                    
                    st.subheader("📍 Optimized Port Order")
                    st.write(" → ".join(res["ordered_ports"]))
                    st.dataframe(pd.DataFrame(res["legs"]), use_container_width=True)
                    
                    st.subheader("🗺️ Multi-Stop Route Map")
                    ms_map = MultiStopOptimizer(analyzer).build_map(res)
                    folium_static(ms_map)
    else:
        # Welcome screen (only if not optimized)
        st.subheader("Welcome to the Marine Route Optimizer!")
        st.markdown("Enter locations and ship details in the sidebar, then click **Optimize Route**.")
        
        sample_start = (37.7749, -122.4194)
        sample_end   = (34.0522, -118.2437)
        m = folium.Map(location=[35.9, -120.3], zoom_start=6, tiles="CartoDB positron")
        sample_sea_route = get_sea_route_coords(sample_start, sample_end)
        folium.PolyLine(locations=sample_sea_route, color='#0057b8', weight=3).add_to(m)
        folium.Marker(sample_start, icon=folium.Icon(color='green', icon='ship', prefix='fa')).add_to(m)
        folium.Marker(sample_end, icon=folium.Icon(color='red', icon='flag', prefix='fa')).add_to(m)
        folium_static(m)

if __name__ == "__main__":
    main()
