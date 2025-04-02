import cv2
import numpy as np
import open3d as o3d
import camera_intrinsic_parameters as params

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def convert_to_opencv_image(o3d_image):
    return cv2.cvtColor(np.array(o3d_image), cv2.COLOR_RGBA2RGB)

def convert_to_open3d_image(cv_image):
    return o3d.geometry.Image(cv_image)

def create_point_cloud_from_rgbd_image(rgbd_image):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(params.camera_params.width, 
                                                  params.camera_params.height, 
                                                  params.camera_params.fx, 
                                                  params.camera_params.fy, 
                                                  params.camera_params.cx, 
                                                  params.camera_params.cy
                                                  )

    logging.info("Converting RGB-D image into point cloud ...")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    return pcd

def compute_widened_contour_line_around_mask(mask, ksize=37):
    mask = mask.astype(np.uint8)  # ensure mask is of type uint8
    outer_mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)  # enlarge mask
    inner_mask = cv2.GaussianBlur(~mask, (ksize, ksize), 0)  # enlarge inverse mask
    new_mask = np.ones_like(mask)
    new_mask[outer_mask == 0] = 0
    new_mask[inner_mask == 0] = 0
    return new_mask


def draw_hough_lines(lines, image):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 5000 * (-b))
            y1 = int(y0 + 5000 * (a))
            x2 = int(x0 - 5000 * (-b))
            y2 = int(y0 - 5000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

import numpy as np
import logging

def create_polygon():
    """
    Creates a bounding box and a selection polygon volume based on the bounding box.

    Returns:
        bbox (o3d.geometry.AxisAlignedBoundingBox): The created bounding box.
        vol (o3d.visualization.SelectionPolygonVolume): The created selection polygon volume.
    """
    logging.debug("Computing bounding box ...")
    # bounding box values for x, y, z
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.45, -1, 0.1), max_bound=(0.45, 0.5, 1.5))
    bbox.color = (0, 1, 1)
    
    # print(np.asarray(bbox.get_box_points()))

    bounding_polygon = np.asarray(bbox.get_box_points())
    logging.debug(f"Bounding polygon: {bounding_polygon}")
    # Create a SelectionPolygonVolume
    vol = o3d.visualization.SelectionPolygonVolume()

    vol.orthogonal_axis = 'Z'
    vol.axis_max = np.max(bounding_polygon[:, 2])
    vol.axis_min = np.min(bounding_polygon[:, 2])

    bounding_polygon[:, 2] = 0

    vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

    return bbox, vol

def compute_point_cloud_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    logging.info(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([0.7, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    return inlier_cloud, plane_model, outlier_cloud

def compute_y(x, line):
    a, b, c = line
    logging.debug(f"Values of the standard form are a={a}, b={b}, c={c}")
    y = (a * x + c) / -b
    logging.debug(f"Computed y value for x={x}: {y}")
    return y

import math

def find_intersection_using_matrix(a1, b1, c1, a2, b2, c2):
    """
    This function finds the intersection point of two lines in standard form using the matrix method.

    Args:
        a1, b1, c1: Coefficients of the first line equation (a1*x + b1*y + c1 = 0).
        a2, b2, c2: Coefficients of the second line equation (a2*x + b2*y + c2 = 0).

    Returns:
        A tuple containing the intersection point (x, y) or None if the lines are parallel or coincident.
    """

    # Convert coefficients to a matrix
    A = np.array([[a1, b1], [a2, b2]])

    # Check for singular matrix (parallel or coincident lines)
    if np.linalg.det(A) == 0:
        return None

    # Convert constants to a column vector
    b = np.array([[-c1], [-c2]])

    # Solve the system of equations using matrix inversion
    x = np.linalg.inv(A).dot(b)

    # Extract the intersection point coordinates
    return float(x[0]), float(x[1])

def calculate_angle_between_lines(a1, b1, c1, a2, b2, c2):
    """
    This function calculates the angle (in degrees) between two lines in standard form.

    Args:
        a1, b1, c1: Coefficients of the first line equation (a1*x + b1*y + c1 = 0).
        a2, b2, c2: Coefficients of the second line equation (a2*x + b2*y + c2 = 0).

    Returns:
        The angle (in degrees) between the two lines.
    """

    # Option 1: Using slopes (if slopes are readily available)
    # m1 = -(a1 / b1)  # assuming b1 is not zero
    # m2 = -(a2 / b2)  # assuming b2 is not zero
    # tan_theta = abs((m1 - m2) / (1 + m1 * m2))

    # Option 2: Using standard form coefficients directly
    tan_theta = abs((a1 * b2 - a2 * b1) / (a1 * a2 + b1 * b2))

    # Convert tan(theta) to degrees
    angle_in_degrees = math.degrees(math.atan(tan_theta))

    return angle_in_degrees

def compute_standard_line_form(x1, y1, x2, y2):
    # Check for division by zero (parallel lines)
    if x1 == x2:
        return None, None  # Parallel lines have undefined slope

    # Calculate slope
    slope = (y2 - y1) / (x2 - x1)

    # Calculate intercept using point-slope form
    intercept = y1 - slope * x1

    a = -slope
    b = -intercept
    c = a * b

    return a, b, c

def check_perpendicular(eq1, eq2, threshold_degrees=5):
    # Extract the coefficients of the lines
    a1, b1, c1 = eq1
    a2, b2, c2 = eq2
    
    # Calculate slopes using standard form (assuming b is not zero to avoid division by zero)
    m1 = -(a1 / b1)
    m2 = -(a2 / b2)

    # Calculate the angle between the lines in degrees
    angle_degrees = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))

    # Check if the angle is within the threshold of 90 degrees
    return abs(angle_degrees - 90) <= threshold_degrees


def draw_lines_through_hough_intersections(intersections, intersection_lines):
    # draw lines through intersections
    for i in range(len(intersections)-1):
        for j in range(i+1, len(intersections)):
            pt1 = intersections[i]
            pt2 = intersections[j]

            # Calculate the direction vector of the line
            extended_pt1, extended_pt2 = compute_line_vector_from_2_points(pt1, pt2)
            angle = np.arctan2(extended_pt2[1] - extended_pt1[1], extended_pt2[0] - extended_pt1[0])
            # logging.info('Line angle: %s', np.degrees(angle))
            line_color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.line(intersection_lines, (int(extended_pt1[0]), int(extended_pt1[1])), (int(extended_pt2[0]), int(extended_pt2[1])), line_color, 5)
            # logging.info('Line color: %s', line_color)
            # Convert the angle to degrees
            angle_degrees = np.degrees(angle)
            # Convert the angle to a string
            angle_text = f"Angle: {angle_degrees:.2f}°"
            # Calculate the position for the text
            text_position = (int((extended_pt1[0] + extended_pt2[0]) / 2) + 300, int((extended_pt1[1] + extended_pt2[1]) / 2))
            text_position = (text_position[0], text_position[1] + (i * 200))  # Adjust the y-coordinate by adding a fixed offset for each line of text
            # Draw the text on the image
            cv2.putText(intersection_lines, angle_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, line_color, 5)
            # cv2.line(intersection_lines, (int(extended_pt1[0]), int(extended_pt1[1])), (int(extended_pt2[0]), int(extended_pt2[1])), (0, 0, 255), 2)
    
    return intersection_lines

def draw_lines_and_compute_bitwise_and(intersection_pixel_coords, zero_matrix, thresholded_edge_image):
    for i in range(len(intersection_pixel_coords)-1):
        for j in range(i+1, len(intersection_pixel_coords)):
            pt1 = intersection_pixel_coords[i][0]
            pt2 = intersection_pixel_coords[j][0]
            # Calculate the direction vector of the line
            extended_pt1, extended_pt2 = compute_line_vector_from_2_points(pt1, pt2)
            line_color = (255, 255, 255)  # White color
            cv2.circle(zero_matrix, (int(pt1[0]), 3072 - int(pt1[1])), 20, line_color, -1)
            cv2.circle(zero_matrix, (int(pt2[0]), 3072 - int(pt2[1])), 20, line_color, -1)
            cv2.line(zero_matrix, (int(extended_pt1[0]), int(extended_pt1[1])), (int(extended_pt2[0]), int(extended_pt2[1])), line_color, 10)

            zero_matrix = zero_matrix.astype(thresholded_edge_image.dtype)

            # compute bitwise and with the defined image
            bitwise_and_image = cv2.bitwise_and(zero_matrix, thresholded_edge_image)

    return bitwise_and_image

            
def compute_line_vector_from_2_points(pt1, pt2):
    # Calculate the direction vector of the line
    direction_vector = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
    # Normalize the direction vector
    normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector)
    # Extend the line by 5000 units
    extended_pt1 = pt1 - 5000 * normalized_direction_vector
    extended_pt2 = pt2 + 5000 * normalized_direction_vector
    # Calculate the angle between the two points
    return extended_pt1, extended_pt2

def compute_line_intersection(lines):
    intersections = []
    for line1 in lines:
        for line2 in lines:
            rho1, theta1 = line1[0]
            rho2, theta2 = line2[0]
            if theta1 != theta2:
                angle1 = np.degrees(theta1)
                angle2 = np.degrees(theta2)
                if abs(angle1 - angle2) > 50:
                    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
                    b = np.array([rho1, rho2])
                    x, y = np.linalg.solve(A, b)
                    intersections.append((x, y))

    return np.array(intersections)

def merge_intersections(intersections, image):
    merged_intersections = []
    for intersection in intersections:
        x, y = intersectionx, y = intersection
        cv2.circle(image, (int(x), int(y)), 10, (255, 255, 0), -1)

        merged = False
        for merged_intersection in merged_intersections:
            merged_x, merged_y = merged_intersection
            distance = np.sqrt((x - merged_x)**2 + (y - merged_y)**2)
            if distance < 15:
                merged_intersection = list(merged_intersection)
                merged_intersection[0] = (x + merged_x) / 2
                merged_intersection = tuple(merged_intersection)
                merged = True
                break
        if not merged:
            merged_intersections.append(intersection)
    intersections = merged_intersections
    return np.array(intersections), image

def transform_image_point_to_pointcloud(point, depth_image, transformation_matrix):
    x, y = point
    z = depth_image[y, x] / 1000
    if z == 0:
        return None
    x = (x - params.camera_params.cx) * z / params.camera_params.fx
    y = (y - params.camera_params.cy) * z / params.camera_params.fy

    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(np.array([[x, y, z]]))

    points.transform(transformation_matrix)

    return points

def transform_plane_points_to_image(points, transformation_matrix, cam_matrix):
    print("len(points): ", len(points[0]))
    if len(points) == 2:
        #add z=0
        points = np.vstack([points, np.array([0, 0, 0])])

    print("type(points): ", type(points))
    if type(points) == np.ndarray:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.paint_uniform_color([1, 0, 0])
    point_cloud.transform(transformation_matrix)
    
    image_points, _ = cv2.projectPoints(np.array(point_cloud.points), np.eye(3,3), np.zeros(3), cam_matrix, np.zeros(5))
    return image_points

def find_perpendicular_plane_ransac(point_cloud, P, N1, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Find a plane whose normal N2 is perpendicular to N1 and closest to point P using RANSAC.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): Input point cloud.
        P (np.ndarray): 3D point [x, y, z] that the plane should be close to.
        N1 (np.ndarray): Reference normal vector [nx, ny, nz] (N2 must be perpendicular to this).
        distance_threshold (float): Max distance for a point to be considered an inlier.
        ransac_n (int): Number of points to sample for RANSAC.
        num_iterations (int): Number of RANSAC iterations.
        
    Returns:
        plane_eq (np.ndarray): Plane equation [a, b, c, d] (ax + by + cz + d = 0).
        N2 (np.ndarray): Plane normal vector (perpendicular to N1).
        inliers (list): Indices of inlier points.
    """
    # Convert point cloud to numpy if needed
    points = np.asarray(point_cloud.points)
    
    # Initialize best plane variables
    best_plane = None
    best_N2 = None
    best_inliers = []
    best_score = -np.inf  # Higher score = better plane
    
    for _ in range(num_iterations):
        # 1. Randomly sample points
        sample_indices = np.random.choice(len(points), size=ransac_n, replace=False)
        samples = points[sample_indices]
        
        # 2. Fit a plane using SVD (Ax + By + Cz + D = 0)
        centroid = np.mean(samples, axis=0)
        centered_samples = samples - centroid
        _, _, Vt = np.linalg.svd(centered_samples)
        N2_candidate = Vt[2, :]  # Last row of Vt is the normal
        
        # 3. Ensure normal points towards P (for consistency)
        if np.dot(N2_candidate, (P - centroid)) < 0:
            N2_candidate = -N2_candidate
        
        # 4. Check perpendicularity constraint (N2 ⊥ N1)
        dot_product = np.dot(N2_candidate, N1)
        if abs(dot_product) > 1e-3:  # Not perpendicular? Project N2 to orthogonal space.
            N2_candidate = N2_candidate - (dot_product) * N1
            N2_candidate = N2_candidate / np.linalg.norm(N2_candidate)
        
        # 5. Compute plane equation: N2 · (X - P) = 0 → N2·X - N2·P = 0 → [a, b, c, d] = [N2, -N2·P]
        d_plane = -np.dot(N2_candidate, P)
        plane_model = np.append(N2_candidate, d_plane)
        
        # 6. Compute inliers (points close to the plane)
        distances = np.abs(points @ N2_candidate + d_plane)
        inliers = np.where(distances < distance_threshold)[0]
        
        # 7. Score the plane (higher = better):
        #    - Maximize inlier count
        #    - Minimize distance from P to plane (|N2·P + d| should be ~0)
        inlier_score = len(inliers)
        distance_score = -np.abs(np.dot(N2_candidate, P) + d_plane)  # Penalize large distances
        total_score = inlier_score + distance_score * 0.1  # Weighted
        
        if total_score > best_score:
            best_score = total_score
            best_plane = plane_model
            best_N2 = N2_candidate
            best_inliers = inliers
    
    if best_plane is None:
        raise ValueError("RANSAC failed to find a valid plane.")
    
    return best_plane, best_N2, best_inliers

def convert_2d_to_3d(points_2d, depth_image, fx, fy, cx, cy):
    points_2d = np.array(points_2d, dtype=np.int32)
    points_3d = np.zeros((len(points_2d), 3), dtype=np.float32)
    
    for i, (x, y) in enumerate(points_2d):
        if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
            z = depth_image[y, x] / 1000.0  # Convert mm to meters (if needed)
            if z <= 0:
                continue
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            points_3d[i] = [x_3d, y_3d, z]
    return points_3d

import math

def distance_3d(point1, point2):
    """Calculate distance using math.dist (Python 3.8+)"""
    return math.dist(point1, point2)


def find_closest_point(points_list, p):
    """
    Find the index of the point in points_list that is closest to point p.
    
    Parameters:
    -----------
    points_list : list
        List of points (each point is a list or array of coordinates)
    p : list or numpy.ndarray
        The query point coordinates
    
    Returns:
    --------
    closest_idx : int
        Index of the closest point in points_list
    """
    # Convert input to numpy arrays
    points_array = np.array(points_list)
    p_array = np.array(p)
    
    # Calculate distances from p to all points
    distances = np.linalg.norm(points_array - p_array, axis=1)
    
    # Find the index of the closest point
    closest_idx = np.argmin(distances)
    
    return closest_idx

def find_rectangle_points(point_cloud, inlier_indices, p1_idx, p2_idx, target_length):
    """
    Find two additional points p3 and p4 that form a rectangle with p1 and p2.
    
    Parameters:
    -----------
    point_cloud : open3d.geometry.PointCloud
        The input point cloud
    inlier_indices : list
        List of indices of points that are inliers of a plane
    p1_idx : int
        Index of the first point in the point cloud
    p2_idx : int
        Index of the second point in the point cloud
    target_length : float
        Target length for the sides of the rectangle
    
    Returns:
    --------
    p3_idx : int
        Index of the third point forming the rectangle
    p4_idx : int
        Index of the fourth point forming the rectangle
    """
    # Extract all points and inlier points
    all_points = np.asarray(point_cloud.points)
    inlier_points = all_points[inlier_indices]
    
    # Extract coordinates of p1 and p2
    p1 = all_points[p1_idx]
    p2 = all_points[p2_idx]
    
    # Verify that p1 and p2 are within the inliers
    if p1_idx not in inlier_indices or p2_idx not in inlier_indices:
        raise ValueError("Points p1 and p2 must be within the plane inliers")
    
    # Get the plane normal by fitting a plane to the inlier points
    pcd_inliers = o3d.geometry.PointCloud()
    pcd_inliers.points = o3d.utility.Vector3dVector(inlier_points)
    plane_model, _ = pcd_inliers.segment_plane(distance_threshold=0.01,
                                              ransac_n=3,
                                              num_iterations=1000)
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    
    # Calculate the vector from p1 to p2
    v_p1p2 = p2 - p1
    p1p2_length = np.linalg.norm(v_p1p2)
    
    # Normalize the p1p2 vector
    v_p1p2_norm = v_p1p2 / p1p2_length
    
    # Compute the perpendicular vector to p1p2 that lies in the plane
    # We use the cross product of the normal and the p1p2 vector
    perpendicular = np.cross(normal, v_p1p2_norm)
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    
    # Scale the perpendicular vector to the target length
    v_perpendicular = perpendicular * target_length
    
    # Calculate the expected positions of p3 and p4
    expected_p3 = p2 + v_perpendicular
    expected_p4 = p1 + v_perpendicular
    
    # Find the closest inlier points to the expected positions
    distances_to_p3 = np.linalg.norm(inlier_points - expected_p3, axis=1)
    distances_to_p4 = np.linalg.norm(inlier_points - expected_p4, axis=1)
    
    closest_to_p3_idx = np.argmin(distances_to_p3)
    closest_to_p4_idx = np.argmin(distances_to_p4)
    
    # Map back to original point cloud indices
    p3_idx = inlier_indices[closest_to_p3_idx]
    p4_idx = inlier_indices[closest_to_p4_idx]
    
    # Verify the rectangle properties
    p3 = all_points[p3_idx]
    p4 = all_points[p4_idx]
    
    # Calculate edge lengths
    len_p1p2 = np.linalg.norm(p2 - p1)
    len_p2p3 = np.linalg.norm(p3 - p2)
    len_p3p4 = np.linalg.norm(p4 - p3)
    len_p4p1 = np.linalg.norm(p1 - p4)
    
    # Calculate angles at corners (dot product of normalized vectors)
    v_p1p2 = (p2 - p1) / np.linalg.norm(p2 - p1)
    v_p2p3 = (p3 - p2) / np.linalg.norm(p3 - p2)
    v_p3p4 = (p4 - p3) / np.linalg.norm(p4 - p3)
    v_p4p1 = (p1 - p4) / np.linalg.norm(p1 - p4)
    
    angle_p1 = np.abs(np.dot(v_p1p2, -v_p4p1))
    angle_p2 = np.abs(np.dot(v_p2p3, -v_p1p2))
    angle_p3 = np.abs(np.dot(v_p3p4, -v_p2p3))
    angle_p4 = np.abs(np.dot(v_p4p1, -v_p3p4))
    
    print(f"Rectangle properties check:")
    print(f"Edge lengths: {len_p1p2:.3f}, {len_p2p3:.3f}, {len_p3p4:.3f}, {len_p4p1:.3f}")
    print(f"Target lengths: original edge {len_p1p2:.3f}, perpendicular edges {target_length:.3f}")
    print(f"Corner angles (closer to 0 is better): {angle_p1:.3f}, {angle_p2:.3f}, {angle_p3:.3f}, {angle_p4:.3f}")
    
    return p3_idx, p4_idx

def visualize_point_with_cloud(point_cloud, point_index):
    """
    Visualizes a point cloud and highlights a specific point using Open3D.

    Parameters:
    - point_cloud: open3d.geometry.PointCloud - The input point cloud
    - point_index: int - Index of the point to highlight
    """

    # Ensure the point cloud is valid
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        print("Error: The input is not a valid Open3D PointCloud object.")
        return

    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)

    # Check if index is within bounds
    if point_index < 0 or point_index >= len(points):
        print("Error: Point index out of range.")
        return

    # Get the target point
    target_point = points[point_index]
    print("Highlighting point at:", target_point)

    # Create a sphere to represent the highlighted point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # Adjust size as needed
    sphere.translate(target_point)  # Move sphere to target point
    sphere.paint_uniform_color([1, 0, 0])  # Red color for visibility

    # Visualize
    o3d.visualization.draw_geometries([point_cloud, sphere])

# def draw_points_and_lines(image, points, color=(255, 0, 0), thickness=2):
#     """
#     Draws 6 points and connects them with lines on a given OpenCV image.
    
#     Parameters:
#     image : np.ndarray
#         OpenCV image (2D image array).
#     points : list of tuples
#         List of four (x, y) coordinates representing points in the image.
#     color : tuple, optional
#         Color of the points and lines in BGR format (default is green).
#     thickness : int, optional
#         Thickness of the lines and circle markers (default is 2).
    
#     Returns:
#     np.ndarray
#         Image with drawn points and lines.
#     """
#     if len(points) != 6:
#         raise ValueError("Exactly 6 points are required.")
    
#     # Draw points
#     for point in points:
#         cv2.circle(image, point, radius=5, color=color, thickness=-1)
    
#     # Draw lines connecting the points
#     for i in range(6):
#         cv2.line(image, points[i], points[(i + 1) % 6], color, thickness)
    
#     return image

def draw_points_and_lines(image, points, color=(0, 255, 0), thickness=2):
    """
    Draws 4 points and connects them with lines on a given OpenCV image.
    
    Parameters:
    image : np.ndarray
        OpenCV image (2D image array).
    points : list of tuples
        List of four (x, y) coordinates representing points in the image.
    color : tuple, optional
        Color of the points and lines in BGR format (default is green).
    thickness : int, optional
        Thickness of the lines and circle markers (default is 2).
    
    Returns:
    np.ndarray
        Image with drawn points and lines.
    """
    if len(points) != 4:
        raise ValueError("Exactly 4 points are required.")
    
    # Draw points
    for point in points:
        cv2.circle(image, point, radius=5, color=color, thickness=-1)
    
    # Draw lines connecting the points
    for i in range(4):
        cv2.line(image, points[i], points[(i + 1) % 4], color, thickness)
    
    return image

def view_2D_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()