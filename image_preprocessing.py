import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
from display_images import display_images
import camera_intrinsic_parameters as params
from helper_functions import find_perpendicular_plane_ransac, convert_2d_to_3d, distance_3d, draw_points_and_lines, view_2D_image

# Load RGB image
image_path = "mnm_boxes_rgb.jpg"
original_rgb_image = o3d.io.read_image("mnm_boxes_rgb.jpg")
cv_image  = cv2.cvtColor(np.array(original_rgb_image), cv2.COLOR_RGBA2RGB)
o3d_image = o3d.geometry.Image(cv_image)

# Load depth image
depth_image = o3d.io.read_image("mnm_boxes_depth.png")
depth_image_cv2 = cv2.imread("mnm_boxes_depth.png", cv2.IMREAD_UNCHANGED)  # Preserves original bit-depth

# Convert to numpy array
depth_array = np.array(depth_image_cv2)

# Create RGB-D image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, depth_image, convert_rgb_to_intensity=False, depth_scale=1000.0)

# Display RGB-D images
# display_images([original_rgb_image, rgbd_image.depth], ["Original Image", "RGB-D Depth Image"])

#####################################################
# Predict segmentation mask for boxes
#####################################################
model = YOLO('all_erlensee_normalcolor_pretrained_mosaic_0_7.pt')  # custom trained YOLOv8n model

prediction_results = model.predict(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
yolo_result = prediction_results[0].plot()
# display_images([yolo_result], ['YOLO Result'])
# save image to file
prediction_results[0].save(filename=f'mask1.png')

numpy_masks = prediction_results[0].masks.cpu().data.numpy()

image_height, image_width = cv_image.shape[:2]

# Create a copy of the original image to overlay the masks
overlay_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB).copy()
# overlay_image = cv_image.copy()
# Loop through each mask and overlay it on the image
# for mask in numpy_masks:
mask = numpy_masks[5]
# Resize the mask to match the size of the original image. Size of each mask/box: 384 x 640
resized_mask = scale_image(mask, cv_image.shape)

# Convert mask to a binary image (0 or 255) to apply it as a mask
binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255  # You can adjust the threshold

# Apply the binary mask onto the overlay image
# Convert binary mask to 3 channels (same as RGB)
binary_mask_colored = cv2.merge([binary_mask, binary_mask, binary_mask])

# Define a transparent color to blend the mask (e.g., green color with some transparency)
mask_color = np.array([0, 255, 0], dtype=np.uint8)  # Green color for the mask

# Apply the mask (with transparency)
alpha = 0.5  # Transparency level (0: fully transparent, 1: fully opaque)
overlay_image = cv2.addWeighted(overlay_image, 1 - alpha, binary_mask_colored, alpha, 0)

# ##############Show only the mask region in the overlay image##########
# # Display the image with the overlay
# plt.figure(figsize=(10, 10))
# plt.imshow(overlay_image)
# plt.axis('off')
# plt.show()

# # Save the final image with overlays
# cv2.imwrite('overlayed_imagex1.jpg', overlay_image)

########### Show both the mask and the contour overlayed on the image ##########
# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw approximated contours to get straight lines
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)  # Approximation factor (tune if needed)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(overlay_image, [approx], -1, (255, 0, 0), 3)  # Blue contour lines

# Display the result
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# Save the final image
cv2.imwrite('mask_and_contour_overlay.jpg', overlay_image)

# ######################################################
# # Find the target point P (bottom edge midpoint)
# ######################################################
# After drawing contours, find the target point P (bottom edge midpoint)
# Find the bottom edge (most likely to be parallel to X-axis and farthest from camera)
bottom_edge = None
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    
    max_y = 0  # Track the lowest edge (highest Y-coordinate)
    min_y = 15000  # Track the highest edge (lowest Y-coordinate)
    # Iterate through all line segments in the contour
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i+1) % len(approx)][0]
        
        # Calculate line properties
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        
        # Check if line is roughly horizontal (within 10 degrees)
        if angle < 10:
            avg_y = (pt1[1] + pt2[1]) / 2
            if avg_y > max_y:  # Select the edge with the largest Y-coordinate (bottom)
                max_y = avg_y
                bottom_edge = (pt1, pt2)

            if avg_y < min_y:  # Select the edge with the smallest Y-coordinate (top)
                min_y = avg_y
                top_edge = (pt1, pt2)
    
if bottom_edge is not None:
    # Calculate midpoint of the bottom edge
    pt1, pt2 = bottom_edge
    P_x = int((pt1[0] + pt2[0]) / 2)
    P_y = int((pt1[1] + pt2[1]) / 2)
    P = (P_x, P_y)
    
    # Draw the point on the image (red circle)
    cv2.circle(overlay_image, P, 10, (0, 0, 255), -1)  # Red circle
    cv2.putText(overlay_image, "P (Bottom Edge Center)", (P_x + 15, P_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Display the result with point P
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# Save the final image
# cv2.imwrite('contour_with_bottom_point_P.jpg', overlay_image)

#####################################################
# Create point cloud
#####################################################
depth_image = cv2.imread('mnm_boxes_depth.png', cv2.IMREAD_UNCHANGED)  # Depth image

# Camera intrinsic parameters (you need to replace these with your actual values)
fx = params.camera_params.fx  # Focal length in pixels along x-axis
fy = params.camera_params.fy  # Focal length in pixels along y-axis
cx = params.camera_params.cx  # Principal point (cx) in pixels (usually the center of the image)
cy = params.camera_params.cy  # Principal point (cy) in pixels (usually the center of the image)

# Initialize the point cloud list
points = []
colors = []

# Initialize the 3D points and 2D image points
object_points = []
image_points = []

# Loop through each pixel in the depth image
height, width = depth_image.shape
for v in range(height):
    for u in range(width):
        # Get the depth value (in meters or millimeters)
        Z = depth_image[v, u] / 1000.0  # Convert depth to meters if it's in millimeters
        
        if Z == 0:  # If the depth is 0 (no valid depth), skip this pixel
            continue
        
        # Compute the 3D coordinates (X, Y, Z) from the depth and RGB image
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        # Get the RGB color for this pixel
        color = cv_image[v, u] / 255.0  # Normalize RGB values to [0, 1]
        
        # Append the point and color to the lists
        points.append([X, Y, Z])
        colors.append(color)

        # Append the 3D object point
        object_points.append([X, Y, Z])
        
        # For simplicity, assume the corresponding 2D image point is the pixel (u, v)
        image_points.append([u, v])

# Convert the list of points and colors into numpy arrays
points = np.array(points)
colors = np.array(colors)

# Convert lists to numpy arrays
object_points = np.array(object_points, dtype=np.float32)
image_points = np.array(image_points, dtype=np.float32)

# Create a point cloud object in Open3D
point_cloud = o3d.geometry.PointCloud()

# Set the points and colors
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# # Visualize the point cloud
print("Displaying 3D point cloud...")
# o3d.visualization.draw_geometries([point_cloud])

if bottom_edge is not None:
    print("Visualizing the bottom edge in 3D...")
    
#####################################################
# Map 2D point P to 3D point cloud
#####################################################
from helper_functions import find_closest_point

# Convert two 2D corner points to 3D coordinates
print("2D top point 1:", top_edge[0])
print("2D top point 2:", top_edge[1])
print("2D p1:", pt1)
print("2D p2:", pt2)

points_2d = np.array([pt1, pt2])
points_3d = convert_2d_to_3d(points_2d, depth_array, fx, fy, cx, cy)

p1 = points_3d[0]
p2 = points_3d[1]

print("3D p1:", p1)
print("3D p2:", p2)
print("Distance of p1 and p2 before mapping onto point cloud", distance_3d(p1, p2))

p1_index = find_closest_point(points, p1)
p2_index = find_closest_point(points, p2)

print("p1_index:", p1_index)
print("p2_index:", p2_index)
print("Distance of p1 and p2 after mapping onto point cloud", distance_3d(points[p1_index], points[p2_index]))

# top_plane_image = draw_points_and_lines(cv_image, [top_edge[0], top_edge[1], pt1, pt2])

# view_2D_image(top_plane_image)

# Ensure we found point P in 2D
if 'P' in locals():

    # --- Method 1: Exact Coordinate Matching ---
    # Find the 3D point whose 2D projection matches P's coordinates
    P_3D = None
    P_3D_idx = None
    min_dist = float('inf')

    # Search through all mapped 2D-3D points
    for i, img_pt in enumerate(image_points):
        dist = np.sqrt((img_pt[0]-P[0])**2 + (img_pt[1]-P[1])**2)
        if dist < min_dist:
            min_dist = dist
            P_3D_idx = i
    
    if P_3D_idx is not None:
        P_3D = points[P_3D_idx]
        print(f"3D coordinates of P: {P_3D}")

#         # Highlight the point in the cloud (green)
#         highlight_colors = np.asarray(point_cloud.colors)
#         highlight_colors[P_3D_idx] = [0, 1, 0]  # Green
#         point_cloud.colors = o3d.utility.Vector3dVector(highlight_colors)

# o3d.visualization.draw_geometries(
#     [point_cloud],
#     window_name="p3 and p4 Visualization",
#     width=1024,
#     height=768,
# )

####################################################
# Map segmentation mask to points in point cloud #
####################################################
# Prepare lists for mask-colored and non-mask points
# mask_points = []
# mask_colors = []
# non_mask_points = []
# non_mask_colors = []

# # Loop through each point and determine if it belongs to the mask
# for i, point in enumerate(points):
#     # Map the 3D point to the corresponding 2D image point
#     x_img, y_img = image_points[i].astype(int)
    
#     # Check if the pixel is within the mask region
#     if resized_mask[y_img, x_img] > 0.5:  # Mask value threshold
#         # Assign mask color (e.g., green) to points in the mask
#         mask_points.append(point)
#         mask_colors.append([0, 1, 0])  # Green color for mask
#     else:
#         # Assign original RGB color to points outside the mask
#         non_mask_points.append(point)
#         non_mask_colors.append(colors[i])

# # Combine mask and non-mask points and colors
# all_points = np.vstack((np.array(mask_points), np.array(non_mask_points)))
# all_colors = np.vstack((np.array(mask_colors), np.array(non_mask_colors)))

# # Create a point cloud object in Open3D
# point_cloud_with_mask = o3d.geometry.PointCloud()
# point_cloud_with_mask.points = o3d.utility.Vector3dVector(all_points)
# point_cloud_with_mask.colors = o3d.utility.Vector3dVector(all_colors)

# # Visualize the point cloud with the mask
# o3d.visualization.draw_geometries([point_cloud_with_mask])

#####################################################
# Extract only the masked point cloud
#####################################################
# Initialize the point cloud list for masked points
masked_points = []
masked_colors = []

# Loop through each pixel in the depth image
for v in range(height):
    for u in range(width):
        # Check if the pixel is within the mask
        if binary_mask[v, u] == 0:  # Skip points outside the mask
            continue
        
        # Get the depth value (in meters or millimeters)
        Z = depth_image[v, u] / 1000.0  # Convert depth to meters if it's in millimeters
        
        if Z == 0:  # If the depth is 0 (no valid depth), skip this pixel
            continue
        
        # Compute the 3D coordinates (X, Y, Z) from the depth and RGB image
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        # Get the RGB color for this pixel
        color = cv_image[v, u] / 255.0  # Normalize RGB values to [0, 1]
        
        # Append the point and color to the lists
        masked_points.append([X, Y, Z])
        masked_colors.append(color)

# Convert the list of points and colors into numpy arrays
masked_points = np.array(masked_points)
masked_colors = np.array(masked_colors)

# Create a point cloud object in Open3D for the masked points
masked_point_cloud = o3d.geometry.PointCloud()
masked_point_cloud.points = o3d.utility.Vector3dVector(masked_points)
masked_point_cloud.colors = o3d.utility.Vector3dVector(masked_colors)

# Visualize the masked point cloud
print("Visualizing the masked point cloud...")
# o3d.visualization.draw_geometries([masked_point_cloud])

# Perform RANSAC plane segmentation on the point cloud of the mask

print('Generating masked plane')
plane_model, plane_1_inliers = masked_point_cloud.segment_plane(
    distance_threshold=0.01,  # Distance threshold for a point to be considered in the plane
    ransac_n=3,              # Number of points to sample for RANSAC
    num_iterations=1000      # Number of RANSAC iterations
)

# Extract the plane equation coefficients
a, b, c, d = plane_model
print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

# Compute the normal vector of the plane
normal_vector = np.array([a, b, c])
normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the vector
print(f"Normal vector of the plane: {normal_vector}")

# #####################################################
# ## Visualize the masked plane from RANSAC method ##
# #####################################################
# # Create a copy of the original point cloud colors
# highlighted_colors = np.asarray(point_cloud.colors).copy()

# # Initialize a list to track which points in the original cloud are inliers
# is_inlier_in_original = np.zeros(len(points), dtype=bool)

# # Create a set of masked points for faster comparison
# masked_points_set = {tuple(point) for point in masked_points}

# # Compare coordinates to find inliers in original cloud
# for i, point in enumerate(points):
#     if tuple(point) in masked_points_set:
#         # This point exists in the masked cloud
#         # Find its index in the masked cloud
#         masked_idx = np.where((masked_points == point).all(axis=1))[0]
#         if len(masked_idx) > 0 and masked_idx[0] in plane_1_inliers:
#             is_inlier_in_original[i] = True

# print("Visualizing the RANSAC plane of top surface...")
# # # Color the inliers green
# # highlighted_colors[is_inlier_in_original] = [0, 1, 0]  # RGB for green

# # # Create visualization cloud
# # highlighted_cloud = o3d.geometry.PointCloud()
# # highlighted_cloud.points = point_cloud.points
# # highlighted_cloud.colors = o3d.utility.Vector3dVector(highlighted_colors)
# # # downsampling for faster visualization
# # highlighted_cloud = highlighted_cloud.voxel_down_sample(voxel_size=0.005)

# # o3d.visualization.draw_geometries([highlighted_cloud],
# #                                  window_name="Original with Inliers",
# #                                  width=1024,
# #                                  height=768)

# #####################################################
# Find orthogonal plane to the masked plane #
#####################################################
plane_2_equation, plane_2_normal, plane_2_inliners = find_perpendicular_plane_ransac(point_cloud, P_3D, normal_vector)

# Extract the plane equation coefficients
a, b, c, d = plane_2_equation
print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

plane_2_normal = plane_2_normal / np.linalg.norm(plane_2_normal)  # Normalize the vector
print(f"Normal vector of the 2nd plane: {plane_2_normal}")
print(np.dot(normal_vector,plane_2_normal))

# # ########################################################
# # # Visualize the masked plane and its orthogonal plane ##
# # ########################################################
# # # Create a copy of the original point cloud colors
# # highlighted_colors = np.asarray(point_cloud.colors).copy()

# # # Initialize a list to track which points in the original cloud are inliers
# # is_inlier_in_original = np.zeros(len(points), dtype=bool)

# # # Create a set of masked points for faster comparison
# # masked_points_set = {tuple(point) for point in masked_points}

# # print('Preparing to visualize masked plane')
# # # Compare coordinates to find inliers in original cloud
# # for i, point in enumerate(points):
# #     if tuple(point) in masked_points_set:
# #         # This point exists in the masked cloud
# #         # Find its index in the masked cloud
# #         masked_idx = np.where((masked_points == point).all(axis=1))[0]
# #         if len(masked_idx) > 0 and masked_idx[0] in plane_1_inliers:
# #             is_inlier_in_original[i] = True
    

# # # Color the inliers green
# # highlighted_colors[is_inlier_in_original] = [0, 1, 0]  # RGB for green

# # # Create visualization cloud
# # highlighted_cloud = o3d.geometry.PointCloud()
# # highlighted_cloud.points = point_cloud.points
# # highlighted_cloud.colors = o3d.utility.Vector3dVector(highlighted_colors)

# # print('Preparing to visualize orthogonal plane')
# # red = np.array([1, 0, 0])  # RGB for red
    
# # # Set inliers to red
# # highlighted_colors[plane_2_inliners] = red
# # # Update the point cloud colors
# # highlighted_cloud.colors = o3d.utility.Vector3dVector(highlighted_colors)

# # print('Downsizing point cloud')
# # # downsampling for faster visualization
# # highlighted_cloud = highlighted_cloud.voxel_down_sample(voxel_size=0.005)

# # print('Displaying point cloud')
# # o3d.visualization.draw_geometries([highlighted_cloud],
# #                                  window_name="Original with Inliers",
# #                                  width=1024,
# #                                  height=768)

# #####################################################
# # Apply the length of the box to the orthogonal plane
# # #####################################################

print("Finding rectangle points...")
def find_rectangle_points(point_cloud, plane_eq, p1_index, p2_index, d1):
    """
    Given a point cloud, a plane equation, and two points (p1, p2), find two additional points (p3, p4)
    such that they form a rectangle in the given plane.

    Parameters:
    - point_cloud: open3d.geometry.PointCloud - The input 3D point cloud.
    - plane_eq: tuple (a, b, c, d) - Plane equation coefficients (ax + by + cz + d = 0).
    - p1_index, p2_index: int - Indices of points p1 and p2 in the point cloud.
    - d1: float - Desired length of the perpendicular sides.

    Returns:
    - (p3_index, p4_index): tuple of indices of the found points.
    """
    a, b, c, d = plane_eq
    points = np.asarray(point_cloud.points)

    # Extract p1 and p2
    p1 = points[p1_index]
    p2 = points[p2_index]
    
    # Compute the direction vector of p1 -> p2
    v1 = p2 - p1
    v1 /= np.linalg.norm(v1)  # Normalize

    # Find a perpendicular vector in the plane
    normal = np.array([a, b, c])  # Plane normal
    v2 = np.cross(v1, normal)  # Cross-product to get a perpendicular vector

    if np.linalg.norm(v2) == 0:
        print("Error: Could not compute perpendicular vector.")
        return None, None

    v2 /= np.linalg.norm(v2)  # Normalize

    # Compute candidate points
    p3_candidate = p2 + d1 * v2
    p4_candidate = p1 + d1 * v2

    # Find nearest points in the cloud
    def find_nearest(point, cloud_points):
        distances = np.linalg.norm(cloud_points - point, axis=1)
        return np.argmin(distances)  # Return index of the nearest point

    p3_index = find_nearest(p3_candidate, points)
    p4_index = find_nearest(p4_candidate, points)

    return p3_index, p4_index


p3_index, p4_index = find_rectangle_points(point_cloud, plane_2_equation, p1_index, p2_index, 0.151)

print("p3_index:", p3_index, points[p3_index])
print("p4_index:", p4_index, points[p4_index])

def project_3d_to_2d(point3d, fx, fy, cx, cy):
    """
    Projects a 3D point to 2D image coordinates using camera intrinsics.
    
    Parameters:
    point3d : list or tuple of three floats
        3D coordinates (x, y, z) in camera space.
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point coordinates (image center offsets).
    
    Returns:
    list
        2D coordinates [u, v] in the image plane.
    """
    x, y, z = point3d
    if z == 0:
        raise ValueError("Depth (z) cannot be zero to avoid division by zero.")
    
    u = (x * fx) / z + cx
    v = (y * fy) / z + cy
    
    return [u, v]

p3_2d = tuple(map(int, project_3d_to_2d(points[p3_index], fx, fy, cx, cy)))
p4_2d = tuple(map(int, project_3d_to_2d(points[p4_index], fx, fy, cx, cy)))

print("2D p1:", pt1)
print("2D p2:", pt2)
print("2D p3:", p3_2d)
print("2D p4:", p4_2d)

# front_plane_image = draw_points_and_lines(cv_image, [top_edge[1], top_edge[0], pt2,p3_2d, p4_2d, pt1])
front_plane_image = draw_points_and_lines(cv_image, [pt1, pt2, p3_2d, p4_2d])

view_2D_image(front_plane_image)
