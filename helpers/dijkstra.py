import heapq
import json
import math
from typing import List, Tuple, Dict, Optional
from helpers.path_analysis import find_sharp_turns, find_nearby_anchors, calculate_area


class Graph:
    def __init__(self, layer_id):
        self.layer_id = layer_id  # Store the layer_id
        self.connections = []  # Store all connections
        self.adjacency_list = {}  # Store connections and their neighbors
        self.portals = []  # Store all portals for the layer
        self.anchor_points = {}  # Store anchor points with their metadata

    def add_connection(self, x1, y1, x2, y2, connection_type, connection_id, layer_id, status):
        """
        Add a connection to the graph.
        """
        connection = {
            'id': connection_id,
            'type': connection_type,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'layer_id': layer_id,  # Store the layer_id in the connection
            'status': status
        }
        self.connections.append(connection)
        self.adjacency_list[connection_id] = []  # Initialize with empty neighbors

    def add_anchor_point(self, anchor_id: str, points: List[Tuple[float, float]]):
        """
        Add an anchor point (POI) to the graph.
        """
        if len(points) > 0:
            area = calculate_area(points)
            centroid = (
                sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points)
            )
            self.anchor_points[anchor_id] = {
                "points": points,
                "area": area,
                "centroid": centroid
            }

    def analyze_path(self, path: List[str], angle_threshold: float = 89,
                    distance_threshold: float = 100,
                    min_distance_between_anchors: float = 200) -> Dict:
        """
        Analyze a path for sharp turns and nearby anchor points.
        Returns a dictionary containing analysis results.
        """
        # Convert connection IDs to points
        path_points = []
        for conn_id in path:
            conn = next((c for c in self.connections if c['id'] == conn_id), None)
            if conn:
                path_points.append((conn['x1'], conn['y1']))
                if conn_id == path[-1]:  # Add final point for last connection
                    path_points.append((conn['x2'], conn['y2']))

        # Find sharp turns
        sharp_turns, gentle_turns = find_sharp_turns(path_points, angle_threshold)

        # Find nearby anchors for sharp turns
        turn_points = [turn[0] for turn in sharp_turns]
        nearby_anchors = find_nearby_anchors(
            turn_points,
            self.anchor_points,
            distance_threshold,
            min_distance_between_anchors
        )

        return {
            "path_points": path_points,
            "sharp_turns": sharp_turns,
            "gentle_turns": gentle_turns,
            "nearby_anchors": nearby_anchors
        }

    def find_intersections(self):
        """
        Find intersections between connections and update adjacency_list.
        Also, add Stop portal mappings to the adjacency list with unidirectional Status control.
        """
        # 1. Normal kesişimlerin bulunması
        for i, conn1 in enumerate(self.connections):
            for j, conn2 in enumerate(self.connections):
                if (
                        i != j
                        and conn1['layer_id'] == conn2['layer_id']
                        and connections_intersect(conn1, conn2)
                ):
                    self.adjacency_list[conn1['id']].append(conn2['id'])

        # 2. Stop portal eşleşmelerini ekleme (Yönlü ve Status kontrolü ile)
        stop_portals = [
            conn for conn in self.connections
            if conn['type'] == 'portal' and conn['id'].startswith('Stop.')
        ]

        for portal in stop_portals:
            base_id = ".".join(portal['id'].split(".")[:2])  # Örneğin 'Stop.3' çıkarılır
            for other_portal in stop_portals:
                if (
                        portal['id'] != other_portal['id']
                        and other_portal['id'].startswith(base_id)
                ):
                    # Yönlü bağlantılar için Status kontrolü
                    if portal['status'] == 'On':
                        self.adjacency_list[portal['id']].append(other_portal['id'])

    def get_all_portals(self):
        # Assuming each connection is a dictionary with a 'type' key
        return [connection for connection in self.connections if connection.get('type') == 'portal']

    def find_connection_by_portal_id(self, portal_id):
        """
        Find the connection corresponding to a portal by matching the portal_id.
        """
        for conn in self.connections:
            if conn['type'] == 'portal' and conn['id'] == portal_id:
                return conn
        return None

    def load_portals_from_connections(self):
        """
        Instead of loading portals from a JSON file, just use self.connections.
        """
        active_portals = [conn for conn in self.connections if
                          conn['type'] == 'portal' and conn['layer_id'] == self.layer_id]



        return active_portals

    def find_closest_portals(self, start_id, layer_id=None, portal_type=None):
        """
        Find the closest 5 portals from the start_id based on Euclidean distance.
        If layer_id is provided, only include portals whose second part of the ID (after the second dot) matches the given layer_id.
        If portal_type is provided, only include portals whose ID starts with the given portal_type.
        If layer_id is None, find the closest portals regardless of layer_id.
        Exclude portals whose ID starts with 'Stop'.
        """
        # Get the connection corresponding to the start ID
        start_connection = next((conn for conn in self.connections if conn['id'] == start_id), None)

        if not start_connection:
            print(f"Start connection with ID {start_id} not found.")
            return None

        # Get the start connection's (x1, y1) coordinates
        start_x1 = start_connection['x1']
        start_y1 = start_connection['y1']

        # Get all active portals
        active_portals = self.load_portals_from_connections()
        if not active_portals:
            print("No active portals found.")
            return []

        # Calculate distance to each portal and store the distance with the portal
        portal_distances = []
        for portal in active_portals:
            # Ensure portal has valid coordinates
            if 'x1' not in portal or 'y1' not in portal:
                print(f"Portal {portal['id']} is missing coordinates.")
                continue

            # Skip portals whose ID starts with 'Stop'
            if portal['id'].startswith('Stop'):
                continue

            # If portal_type is provided, filter by portal ID's prefix
            if portal_type and not portal['id'].startswith(portal_type):
                continue

            portal_x1 = portal['x1']
            portal_y1 = portal['y1']

            # Calculate Euclidean distance from start to portal
            distance = math.sqrt((portal_x1 - start_x1) ** 2 + (portal_y1 - start_y1) ** 2)

            # Extract the part of the portal ID after the second dot
            portal_parts = portal['id'].split('.')
            if len(portal_parts) > 2:
                portal_layer_id = portal_parts[2]  # The part after the second dot (third part of ID)

                # Only add the portal if layer_id matches, or if layer_id is not provided
                if layer_id is None or portal_layer_id == str(layer_id):
                    portal_distances.append((portal, distance))
            else:
                print(f"Portal ID {portal['id']} is malformed.")

        # Sort the portals by distance to the start point and get the closest 5
        portal_distances.sort(key=lambda x: x[1])
        closest_portals = [portal for portal, _ in portal_distances[:5]]

        if not closest_portals:
            if layer_id is not None:
                print(f"No portals found on layer {layer_id}.")
            elif portal_type is not None:
                print(f"No portals found with type {portal_type}.")
            else:
                print("No portals found.")

        return closest_portals

    def find_correct_passage_portal(self, valid_portals, target_layer_id):
        """
        Find the corresponding portal for each portal in the valid_portals list.
        A portal's corresponding portal must:
        1. Have the same first two parts in its ID.
        2. Have a layer_id that matches the target_layer_id.
        """
        corresponding_portals = []

        # Loop through each portal in valid_portals
        for portal in valid_portals:
            portal_id_parts = portal['id'].split('.')
            portal_first_two_parts = '.'.join(portal_id_parts[:2])  # First two parts of the ID

            # Now, we need to find a portal where the ID's first two parts match and the last part matches the target layer ID
            matching_portal = None
            for other_portal in self.load_portals_from_connections():  # Assuming you are loading all portals again here
                other_portal_id_parts = other_portal['id'].split('.')
                other_portal_first_two_parts = '.'.join(other_portal_id_parts[:2])

                # Check if first two parts match and the last part equals target_layer_id
                if other_portal_first_two_parts == portal_first_two_parts and other_portal_id_parts[-1] == str(
                        target_layer_id):
                    matching_portal = other_portal
                    break

            # If a matching portal is found, add it to the corresponding_portals list
            if matching_portal:
                corresponding_portals.append(matching_portal)
            else:
                print(f"No corresponding portal found for portal {portal['id']} with target layer {target_layer_id}.")

        return corresponding_portals



def connections_intersect(conn1, conn2, tolerance=0.5):
    """
    Check if two connections intersect at any of their endpoints with a tolerance.
    """
    c1_points = [(conn1['x1'], conn1['y1']), (conn1['x2'], conn1['y2'])]
    c2_points = [(conn2['x1'], conn2['y1']), (conn2['x2'], conn2['y2'])]

    for point1 in c1_points:
        for point2 in c2_points:
            # Check if the difference between points is within the tolerance range
            if abs(point1[0] - point2[0]) <= tolerance and abs(point1[1] - point2[1]) <= tolerance:
                return True
    return False


def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def heuristic(graph, conn_id, goal_id):
    """
    Calculate the heuristic (Euclidean distance) between two connections.
    """
    conn1 = next(conn for conn in graph.connections if conn['id'] == conn_id)
    conn2 = next(conn for conn in graph.connections if conn['id'] == goal_id)
    return euclidean_distance((conn1['x2'], conn1['y2']), (conn2['x1'], conn2['y1']))


def distance_between(graph, conn1_id, conn2_id):
    """
    Calculate the actual distance between two connected connections.
    """
    conn1 = next(conn for conn in graph.connections if conn['id'] == conn1_id)
    conn2 = next(conn for conn in graph.connections if conn['id'] == conn2_id)
    return euclidean_distance((conn1['x2'], conn1['y2']), (conn2['x1'], conn2['y1']))


def dijkstra_connections(graph, start_id, goal_id):
    """
    Dijkstra's algorithm implementation for finding shortest path between connections.
    """
    open_set = []
    heapq.heappush(open_set, (0, start_id))  # (distance, connection_id)
    came_from = {}  # To reconstruct the path
    g_score = {conn['id']: float('inf') for conn in graph.connections}
    g_score[start_id] = 0

    visited_paths = set()  # Use a set for faster lookup

    while open_set:
        current_distance, current_id = heapq.heappop(open_set)

        if current_id == goal_id:
            # Path found, reconstruct it
            path = []
            while current_id in came_from:
                path.append(current_id)
                current_id = came_from[current_id]
            path.append(start_id)
            path.reverse()
            return path

        # Add the current path to visited paths
        visited_paths.add(current_id)

        # Check all neighbors of current connection
        for neighbor_id in graph.adjacency_list[current_id]:
            if neighbor_id in visited_paths:
                continue

            # Calculate tentative g_score
            tentative_g_score = g_score[current_id] + distance_between(graph, current_id, neighbor_id)

            if tentative_g_score < g_score[neighbor_id]:
                # This path is better than any previous one
                came_from[neighbor_id] = current_id
                g_score[neighbor_id] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor_id))

    # No path found, print visited paths for debugging
    print("No path found. Visited paths:")
    for path in visited_paths:
        print(path)
    return None


def find_matching_portals(graph, portal_id, layer_id):
    """
    Verilen bir portal ID'si için, belirtilen grafikteki eşdeğer portalları bulur ve
    ikinci noktadan sonraki kısmı 'target_value' olanları döndürür.

    Parametreler:
    graph (Graph): Portalların bulunduğu grafik.
    portal_id (str): Eşleşmesini bulmak istediğimiz portal ID'si.
    target_value (str): İkinci noktadan sonraki kısmın eşleşmesi gereken değer.

    Döndürülen değer:
    matching_portals (list): Belirtilen grafikteki eşleşen portal ID'leri.
    """
    matching_portals = []  # Eşleşen portalları saklamak için liste

    # Portal kimliğinin temel kısmını al (örneğin, "Elev.5" gibi)
    portal_base = '.'.join(portal_id.split('.')[:2])  # İlk iki kısmı almak için değiştirdim


    # Verilen grafikteki portalları kontrol et

    for portal in graph.get_all_portals():  # get_all_portals() fonksiyonunun doğru olduğundan emin ol
        portal_other_id = portal['id']


        # Eğer portal ID'si, portal_base'ı içeriyorsa
        if portal_base in portal_other_id:  # Portal ID'si içinde portal_base'ı kontrol et

            # İkinci noktadan sonraki kısmı target_value ile karşılaştır
            parts = portal_other_id.split('.')  # Portal ID'sini parçalara ayır
            if len(parts) > 2 and parts[2] == str(layer_id):  # İkinci noktadan sonraki kısım target_value mı?
                print(f"Adding portal: {portal}")  # Check which portal is being added
                print(f"Matching portal found: {portal_other_id}")
                matching_portals.append(portal),

    return matching_portals