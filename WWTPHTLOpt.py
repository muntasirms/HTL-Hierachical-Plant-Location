import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import pandas as pd

# --- Set Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -- constraints

def compute_lambda(current_step, total_steps, lambda_init, lambda_final, schedule_type='log'):
    """Compute scheduled lambda value based on progress. Effectively a scheduler for hardness of the constraint. By initializing constraints as soft and slowly enforcing their hardness,
    the solver is more likely to find a global optimum.
    """
    progress = min(1.0, current_step / total_steps)

    if schedule_type == 'linear':
        lambda_val = lambda_init + (lambda_final - lambda_init) * progress
    elif schedule_type == 'exp':
        lambda_val = lambda_init * (lambda_final / lambda_init) ** progress
    elif schedule_type == 'log':
        log_interp = math.log10(lambda_init) + (math.log10(lambda_final) - math.log10(lambda_init)) * progress
        lambda_val = 10 ** log_interp
    elif schedule_type == 'sigmoid':
        lambda_val = lambda_init + (lambda_final - lambda_init) * (1 / (1 + math.exp(-10 * (progress - 0.5))))
    else:
        lambda_val = lambda_final  # Default to final value

    return lambda_val

def constraint_penalty_with_schedule(
        parameters,
        constraint_function,
        min_val,
        max_val,
        lambda_init=0.1,
        lambda_final=100000.0,
        current_step=0,
        total_steps=1000,
        schedule_type='log'
):
    """
    Apply a constraint penalty on a function of parameters with a ramping schedule

    Parameters:
    - parameters: List of tensors that are being optimized
    - constraint_function: Function that takes parameters and returns a tensor
    - min_val: Minimum allowed value for the function output
    - max_val: Maximum allowed value for the function output
    - lambda_init: Initial penalty weight
    - lambda_final: Final penalty weight
    - current_step: Current optimization step
    - total_steps: Total number of optimization steps
    - schedule_type: Type of schedule ('linear', 'exp', 'log', 'sigmoid')

    Returns:
    - Loss term representing constraint violation with scheduled weight
    """
    # Compute the ramping factor (between 0 and 1)
    progress = min(1.0, current_step / total_steps)

    # Apply the chosen schedule
    if schedule_type == 'linear':
        lambda_val = lambda_init + (lambda_final - lambda_init) * progress
    elif schedule_type == 'exp':
        lambda_val = lambda_init * (lambda_final / lambda_init) ** progress
    elif schedule_type == 'log':
        # Log interpolation as in your example
        log_interp = math.log10(lambda_init) + (math.log10(lambda_final) - math.log10(lambda_init)) * progress
        lambda_val = 10 ** log_interp
    elif schedule_type == 'sigmoid':
        # Sigmoid schedule (slower at beginning and end, faster in middle)
        lambda_val = lambda_init + (lambda_final - lambda_init) * (1 / (1 + math.exp(-10 * (progress - 0.5))))
    else:
        lambda_val = lambda_final  # Default to final value

    # Apply the function to get the values we want to constrain
    function_output = constraint_function(*parameters)

    # Calculate violations
    below_min = torch.relu(min_val - function_output)
    above_max = torch.relu(function_output - max_val)

    # Compute penalty
    penalty = below_min ** 2 + above_max ** 2

    # Return weighted sum with scheduled lambda
    return lambda_val * torch.sum(penalty)

def soft_or_penalty_with_schedule(
        penalties,
        current_step,
        total_steps,
        lambda_init=0.1,
        lambda_final=100.0,
        schedule_type='log',
        beta=10.0
):
    """
    Numerically stable version of soft-or penalty with schedule.
    Maintains original structure while preventing NaN/Inf errors.
    """
    # 1. Input validation and preprocessing
    valid_penalties = [p for p in penalties if p is not None and not torch.isnan(p)]
    if not valid_penalties:
        return torch.tensor(0.0, device='cuda', requires_grad=True)

    penalties_tensor = torch.stack(valid_penalties)

    # 2. Normalize penalties to reasonable range (prevents overflow)
    mean_mag = torch.mean(torch.abs(penalties_tensor.detach())) + 1e-7
    penalties_normalized = penalties_tensor / mean_mag

    # 3. Compute scheduled parameters with safeguards
    beta = min(max(beta, 1e-3), 50.0)  # Clamp beta to [1e-3, 50]
    lambda_val = compute_lambda(
        current_step,
        total_steps,
        max(lambda_init, 1e-7),  # Prevent zero init
        min(lambda_final, 1e7),  # Prevent excessive final
        schedule_type
    )

    # 4. Numerically stable soft-min calculation
    shifted = -beta * penalties_normalized
    max_val = shifted.max()  # For numerical stability

    # Stabilized exponential terms
    exp_terms = torch.exp(shifted - max_val)
    sum_exp = torch.clamp(torch.sum(exp_terms), min=1e-20)  # Prevent log(0)

    # Final computation
    soft_min_penalty = (torch.log(sum_exp) + max_val) / -beta

    # 5. Apply scheduled weight and denormalize
    return (lambda_val * mean_mag) * soft_min_penalty

def soft_and_penalty_with_schedule(
        penalties,
        current_step,
        total_steps,
        lambda_init=0.1,
        lambda_final=100000.0,
        schedule_type='log',
        beta=10.0
):
    """
    Combine an arbitrary number of penalties using a soft AND with a ramping schedule.

    Parameters:
    - penalties: List of individual penalty tensors.
    - current_step: Current optimization step.
    - total_steps: Total number of optimization steps.
    - lambda_init: Initial penalty weight.
    - lambda_final: Final penalty weight.
    - schedule_type: Type of schedule ('linear', 'exp', 'log', 'sigmoid').
    - beta: Soft-max sharpness parameter (higher beta -> closer to true maximum).

    Returns:
    - Overall penalty: Scheduled weight times the soft-max of the individual penalties.
    """
    # Get scheduled lambda value (ramping penalty weight)
    beta = compute_lambda(current_step, 5000, 1e-7, 10, schedule_type)
    beta = min(max(beta, 1e-3), 50.0)  # Clamp beta to [1e-3, 50]

    lambda_val = compute_lambda(current_step, total_steps, lambda_init, lambda_final, schedule_type)

    # Convert list of penalties to a tensor
    # for i, penalty in enumerate(penalties):
    #     penalties[i]  = (penalty - penalty.mean()) / (penalty.std() + 1e-7)

    penalties_tensor = torch.stack(penalties)

    # Compute soft-max: a smooth approximation to the maximum over the penalties
    max_val = torch.max(beta * penalties_tensor)
    stable_exp = torch.exp(beta * penalties_tensor - max_val)
    soft_max_penalty = (torch.log(torch.sum(stable_exp)) + max_val) / beta
    # soft_max_penalty = 1.0 / beta * torch.log(torch.sum(torch.exp(beta * penalties_tensor)))

    # Multiply by the scheduled lambda weight
    return lambda_val * soft_max_penalty




# --- Helper: Haversine distance function ---
def haversine_distance(feed_coords, plant_coords, R=6371.0):
    """
    Compute the Haversine distance in kilometers between each feedstock source and plant.
    feed_coords: Tensor of shape (n, 2) with lat, lon in degrees.
    plant_coords: Tensor of shape (m, 2) with lat, lon in degrees.
    Returns a tensor of shape (n, m) with distances.
    """
    # Convert degrees to radians
    feed_rad = feed_coords * math.pi / 180.0  # shape: (n, 2)
    plant_rad = plant_coords * math.pi / 180.0  # shape: (m, 2)

    # Expand dimensions to broadcast properly:
    # feed_rad: (n, 1, 2) and plant_rad: (1, m, 2)
    feed_rad = feed_rad.unsqueeze(1)  # (n, 1, 2)
    plant_rad = plant_rad.unsqueeze(0)  # (1, m, 2)

    # Differences in latitude and longitude (in radians)
    dlat = plant_rad[..., 0] - feed_rad[..., 0]  # (n, m)
    dlon = plant_rad[..., 1] - feed_rad[..., 1]  # (n, m)

    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(feed_rad[..., 0])
        * torch.cos(plant_rad[..., 0])
        * torch.sin(dlon / 2) ** 2
    )
    a = torch.clamp(a, min=1e-7, max=1.0 - 1e-7)  # Avoid edge cases
    sqrt_a = torch.sqrt(a)
    sqrt_1ma = torch.sqrt(1.0 - a + 1e-7)  # Prevent sqrt(negative)
    c = 2 * torch.atan2(sqrt_a, sqrt_1ma)
    distance = R * c  # in kilometers

    return distance  # shape: (n, m)


def region_penalty(plant_coords, forbidden_regions, penalty_strength=1e7):
    penalty = 0.0
    for plant in plant_coords:
        for region in forbidden_regions:
            if region["type"] == "circle":
                center = torch.tensor(region["center"], device=device)
                dist = haversine_distance(plant.unsqueeze(0), center.unsqueeze(0))
                penalty += penalty_strength * torch.relu(region["radius_km"] - dist) ** 2

            elif region["type"] == "rectangle":
                lat_pen = torch.relu(region["min_lat"] - plant[0]) + torch.relu(plant[0] - region["max_lat"])
                lon_pen = torch.relu(region["min_lon"] - plant[1]) + torch.relu(plant[1] - region["max_lon"])
                penalty += penalty_strength * (lat_pen + lon_pen)
    return penalty


# --- Model / Optimization Setup ---

# Dummy data sizes
n = 4283  # number of feedstock sources
m = 250  # number of candidate plants
next_hierarchy_m = 10 # number of next hierarchy plants
k = 5 # number of mass components to track

# Generate dummy feedstock data and assign them to device:
csv_path = "WWTPs.csv"  # Path to your CSV
feed_data = pd.read_csv(csv_path, header=None)

longitudes = torch.tensor(feed_data.iloc[:, 0].values, dtype=torch.float32, device=device)
latitudes = torch.tensor(feed_data.iloc[:, 1].values, dtype=torch.float32, device=device)
feed_amount = torch.tensor(feed_data.iloc[:, 2].values, dtype=torch.float32, device=device)

feed_coords = torch.stack((longitudes, latitudes), dim=1)

tipping_fee = torch.empty(n, device=device).uniform_(-75, -25)  # Tipping fee per unit feed
feed_compositions = torch.rand(n,k, device=device)
feed_compositions /= feed_compositions.sum(dim=1,keepdim=True)


forbidden_regions = [
    # Series of circles along river coordinates
    {"type": "circle", "center": (0, 0), "radius_km": 1500},
    # {"type": "circle", "center": (35.2, 90.3), "radius_km": 1500},
    # {"type": "circle", "center": (35.3, 90.4), "radius_km": 1500},
    # ... 20 more circles along the river path
]

# --- Heuristic Initialization for Plant Locations ---
# Sort feedstock sources by feed_amount (in descending order)
sorted_indices = torch.argsort(feed_amount, descending=True)
# Use the top m feedstock locations as the initial candidate plant positions.
plant_coords_init = feed_coords[sorted_indices][:m].clone()
# Optional: add a tiny random jitter if desired
# plant_coords_init += torch.randn_like(plant_coords_init) * 0.01

# Create plant coordinates as optimizable parameters, and move them to device.
plant_coords = torch.nn.Parameter(plant_coords_init.to(device))
next_hierarchy_plant_coords = feed_coords[sorted_indices][:next_hierarchy_m].clone()
next_hierarchy_plant_coords = torch.nn.Parameter(next_hierarchy_plant_coords.to(device))
next_hierarchy_tipping_fee = torch.empty(m, device=device).uniform_(0.5, 2.0)  # Tipping fee per unit feed


# --- Hyperparameters for Cost Functions ---
transport_cost_factor = 0.059/1.61/264*1000000  # cost per unit feed per km (converted from $/m3/mi to $/mmgal/km with dewatering rate from table 4 of https://www.sciencedirect.com/science/article/pii/S096585641500018X
orphan_penalty = 50  # cost per unit feed if not delivered ("orphan" option)
capital_cost_coef = -0.68  # coefficient in capital cost function
capital_cost_exponent = 2  # exponent for capital cost (economies of scale)
revenue_coef = 1000*.4*45/38*3.5  # revenue coefficient
revenue_exponent = 1  # exponent for revenue calculation

# Setup assignment logits:
# Each feedstock source can distribute feed among m candidate plants and one "orphan" option.
assignment_logits = torch.nn.Parameter(torch.zeros(n, m + 1, device=device))
next_hierarchy_assignment_logits = torch.nn.Parameter(torch.zeros(m, next_hierarchy_m+1, device=device))

# Define the optimizer (optimize both plant_coords and assignment_logits)
optimizer = torch.optim.Adam([plant_coords, assignment_logits, next_hierarchy_plant_coords, next_hierarchy_assignment_logits], lr=.01)
# The scheduler will reduce the learning rate if the cost does not decrease for 'patience' epochs.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=500, factor=0.9, verbose=True
)

# --- Optimization Loop ---
num_epochs = 500000  # Adjust number of epochs as needed
previous_cost = None

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Compute soft assignments using softmax (first m: candidate plants; last: orphan).
    assignments = F.softmax(assignment_logits, dim=1)  # shape: (n, m+1)
    next_hierarchy_assignments = F.softmax(next_hierarchy_assignment_logits, dim=1) # shape: (m, next_hierarchy_m+1)

    # --- Compute Delivery Cost ---
    distances = haversine_distance(feed_coords, plant_coords)  # shape: (n, m)
    cost_delivery = feed_amount.unsqueeze(1) * (tipping_fee.unsqueeze(1) + transport_cost_factor * distances)  # (n, m)
    cost_orphan = (feed_amount * orphan_penalty).unsqueeze(1)  # (n, 1)
    cost_matrix = torch.cat([cost_delivery, cost_orphan], dim=1)
    delivery_cost = torch.sum(assignments * cost_matrix)

    # --- Compute Capital Cost ---
    plant_loads = torch.sum(feed_amount.unsqueeze(1) * assignments[:, :m], dim=0)
    capital_cost = capital_cost_coef * torch.sum(plant_loads ** capital_cost_exponent)

    # --- Compute Revenue ---
    total_delivered_feed = torch.sum(feed_amount * torch.sum(assignments[:, :m], dim=1))
    revenue = revenue_coef * (total_delivered_feed ** revenue_exponent)

    # delivery cost 2
    next_hierarchy_distances = haversine_distance(plant_coords, next_hierarchy_plant_coords) # shape (m, next_hierarchy_m)
    next_hierarchy_cost_delivery = plant_loads.unsqueeze(1) * ( transport_cost_factor * next_hierarchy_distances)  #  (m, next_hierarchy_m)
    next_hierarchy_cost_orphan = (plant_loads * orphan_penalty).unsqueeze(1)  # (m, 1)
    next_hierarchy_cost_matrix = torch.cat([next_hierarchy_cost_delivery, next_hierarchy_cost_orphan], dim=1)
    next_hierarchy_delivery_cost = torch.sum(next_hierarchy_assignments * next_hierarchy_cost_matrix)

    # capital cost 2
    next_hierarchy_plant_loads = torch.sum(plant_loads.unsqueeze(1) * next_hierarchy_assignments[:, :next_hierarchy_m], dim=0)
    next_hierarchy_capital_cost = capital_cost_coef * torch.sum(next_hierarchy_plant_loads ** capital_cost_exponent)

    # revenue 2
    next_hierarchy_total_delivered_feed = torch.sum(plant_loads * torch.sum(next_hierarchy_assignments[:, :next_hierarchy_m], dim=1))
    next_hierarchy_revenue = revenue_coef * (next_hierarchy_total_delivered_feed ** revenue_exponent)

    # split streams
    # stream_compositions = assignments*feed_compositions # compositions of each material k from each feedstock source i that are sent to plants of hierarchy 1
    # plant_loads_by_composition = 0 # torch.sum(each column of the stream compositions tensor - i.e. sum all assignments to this plant of component k. each plant will then have a tensor of length k with exact mass flows to that plant)
    # plant_output_transform = None # this is the hardest part. some way to transform total mass input to plant into defined number of outputs with each output stream (plus waste) having a user defined composition mass balance constrained. needs to constrain mass balance, allow user defined number and composition of streams, and still allow extensible reading of the output composition tensor downstream
    # plant_outputs = None # plant output streams split by composition

    # next layer of assignment logits need to account for all output streams of all plants while conserving mass...
    # revenue, capital cost, operating cost can now be a function of each stream produced? or
    # use einsum to track mass balances of each material component at each nodes
    # can then calculate sub/helper tensors that also enable constraints
    # [feedstock source, plant, mass flow of each material]

    #     [plant, input mass, composition of mass flow]

    #     [plant, next_plant, output stream, mass flow of each material]

    #     [next_plant, input mass, composition of mass flow]
    delivery_costs = torch.sum(cost_delivery,
                               dim=0)  # torch.sum(assignments * cost_matrix, dim=1) * feed_amount * torch.sum(assignments[:, :m], dim=1)

    revenues = revenue_coef * torch.sum(assignments[:, :m] * feed_amount.unsqueeze(1), dim=0) ** revenue_exponent

    capital_costs = capital_cost_coef * torch.sum(assignments[:, :m] * feed_amount.unsqueeze(1),
                                                  dim=0) ** capital_cost_exponent

    # test constraint

    # space_constraint = constraint_penalty_with_schedule([plant_coords[...,:]], lambda x: x, -30, -15, current_step=epoch, total_steps=3000)
    # space_constraint2 = constraint_penalty_with_schedule([plant_coords[...,:]], lambda x: x, -70, -65, current_step=epoch, total_steps=3000)
    # space_constraint3 = constraint_penalty_with_schedule([next_hierarchy_plant_coords], lambda x: x, -50, -45, current_step=epoch, total_steps=3000)
    # space_constraint = constraint_penalty_with_schedule([plant_coords], lambda x: x, 10, 15, current_step=epoch, total_steps=3000)
    cost_constraint = constraint_penalty_with_schedule([revenue, delivery_cost, capital_cost], lambda x,y,z: z, 0, 1000000, current_step=epoch, total_steps=3000)
    orphan_constraint = constraint_penalty_with_schedule([cost_orphan], lambda x: x, -1000000000000000000, 0, current_step=epoch, total_steps=3000)


    # penalties = [space_constraint, space_constraint2, space_constraint3, cost_constraint]

    # plantLocationPenalties = [space_constraint2, space_constraint]
    # plantPenalties = soft_or_penalty_with_schedule(plantLocationPenalties, current_step=epoch, total_steps=1000, lambda_init=.1, lambda_final=10000.,
    #                                schedule_type='log')
    # nextPlantLocationPenalties = [space_constraint3]
    #
    #
    #
    # overall_penalty = soft_and_penalty_with_schedule([cost_constraint], current_step=epoch, total_steps=1000, lambda_init=.1, lambda_final=10000.,
    #                                                 schedule_type='log')

    # --- Overall Objective ---
    overall_cost = delivery_cost + capital_cost - revenue + next_hierarchy_delivery_cost + next_hierarchy_capital_cost - next_hierarchy_revenue + cost_constraint #+ overall_penalty # + region_penalty(plant_coords, forbidden_regions)

    overall_cost.backward()
    optimizer.step()

    # Scheduler step: we pass the overall cost value to determine if we need to adjust the learning rate.
    # scheduler.step(overall_cost.item())

    # Monitor progress (and check if cost improvement is small)
    if epoch % 50 == 0 or epoch == num_epochs - 1:
        current_lr = optimizer.param_groups[0]['lr']
        print("Iteration {:3d}, cost: {:.4f}, learning rate: {:.6f}, NPV: {:.4f}".format(epoch, overall_cost.item(), current_lr, -(overall_cost-orphan_constraint)))

    # Optionally, check convergence by comparing with previous cost.

    if previous_cost is not None and abs(previous_cost - overall_cost.item()) < 1e-7 and epoch > 5000:
        print(f"Convergence tolerance reached at epoch {epoch}.")
        break
    previous_cost = overall_cost.item()

# --- Extract Final Results ---
final_plant_positions = plant_coords.detach().to("cpu")
final_assignments = assignments.detach().to("cpu")
final_decisions = torch.argmax(final_assignments, dim=1)

next_hierarchy_final_plant_positions = next_hierarchy_plant_coords.detach().to("cpu")
next_hierarchy_final_assignments = next_hierarchy_assignments.detach().to("cpu")

print("\nFinal Candidate Plant Positions (lat, lon):")
print(final_plant_positions)
print("\nFinal Assignment Decision for each feedstock source (0..m-1 delivered to plant; m = orphan):")
print(final_decisions)

# --- Plotting ---

# Convert tensors to numpy arrays for plotting.
feed_coords_np = feed_coords.detach().cpu().numpy()  # (n, 2)
plant_coords_np = final_plant_positions.numpy()       # (m, 2)
feed_amount_np = feed_amount.detach().cpu().numpy()   # (n,)
assignments_np = final_assignments.numpy()            # (n, m+1)


next_hierarchy_plant_coords_np = next_hierarchy_final_plant_positions.numpy()
next_hierarchy_assignments_np = next_hierarchy_final_assignments.numpy()
plant_loads_np = plant_loads.detach().cpu().numpy()   # (n,)


plt.figure(figsize=(10,8))
# Plot feedstock sources (blue) and candidate plants (red).
plt.scatter(feed_coords_np[:, 0], feed_coords_np[:, 1], color='blue', label='Feedstock Sources', alpha=0.6)
plt.scatter(plant_coords_np[:, 0], plant_coords_np[:, 1], color='red', label='Candidate Plants', s=100)
plt.scatter(next_hierarchy_plant_coords_np[:, 0], next_hierarchy_plant_coords_np[:, 1], color='green')

# --- Plot Weighted Chords ---
# For each feedstock source to each candidate plant link (ignoring the orphan column)
weights = []
n_val, mplus = assignments_np.shape
for i in range(n_val):
    for j in range(m):  # only candidate plants, not the orphan column
         delivered_feed = feed_amount_np[i] * assignments_np[i, j]
         weights.append(delivered_feed)
max_weight = max(weights) if weights else 1.0

# Draw chords (green lines) with a line width scaled with delivered feed.
for i in range(n_val):
    for j in range(m):
         w = feed_amount_np[i] * assignments_np[i, j]
         # Skip very low deliveries to reduce clutter.
         if w < 1e-3:
             continue
         # Scale linewidth: here, min=0.5 and max=4.5 (adjust as needed)
         lw = 0.5 + 4 * (w / max_weight)
         # Coordinates: note that longitude is plotted on x and latitude on y.
         x_vals = [feed_coords_np[i, 1], plant_coords_np[j, 1]]
         y_vals = [feed_coords_np[i, 0], plant_coords_np[j, 0]]
         # plt.plot(y_vals, x_vals, color='green', linewidth=lw, alpha=0.7)

n_val, mplus = next_hierarchy_assignments_np.shape

# Draw chords (green lines) with a line width scaled with delivered load.
for i in range(n_val):
    for j in range(next_hierarchy_m):

        # Scale linewidth: here, min=0.5 and max=4.5 (adjust as needed)

        # Coordinates: note that longitude is plotted on x and latitude on y.
        x_vals = [plant_coords_np[i, 1], next_hierarchy_plant_coords_np[j, 1]]
        y_vals = [plant_coords_np[i, 0], next_hierarchy_plant_coords_np[j, 0]]
        # plt.plot(y_vals, x_vals, color='blue', linewidth=0.01, alpha=0.7)

plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Feedstock Sources, Candidate Plants, and Weighted Connections")
plt.legend()
plt.show()

