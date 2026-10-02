#!/usr/bin/env python3
"""
alns_core.py — operators and evaluation functions for the PC-MT-EVRP-RT-D ALNS.

Lifted verbatim from cell 1 of main_ALNS.ipynb. The function bodies are
UNCHANGED so that this module reproduces the notebook exactly; main_ALNS.ipynb
remains the reference implementation to diff against.

Instance-level constants
------------------------
Several functions read module-level names that were notebook globals:

    trip_distance             -> traveling_time
    check_capacity_g          -> customer_loads
    objective_function_global -> C
    objective_function_value  -> eta, recharging_rate, f_recharging,
                                 release_time, time_windows

Call `bind_instance(...)` once per instance before using anything else. This
mirrors notebook semantics exactly, at the cost of the module holding one
instance at a time: do NOT evaluate two different instances concurrently in
the same process. Run separate processes for parallel experiments.

    from alns_core import bind_instance, objective_function_global
    bind_instance(C=C, traveling_time=traveling_time, customer_loads=customer_loads,
                  time_windows=time_windows, release_time=release_time,
                  eta=eta, recharging_rate=recharging_rate, f_recharging=f_recharging)
"""

import copy
import heapq
import itertools
import math
import random
from copy import deepcopy
from itertools import chain, combinations

import numpy as np
import pandas as pd

# Only plot_vrp_multitrip needs these; importing lazily keeps the module usable
# on headless machines without a matplotlib backend configured.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover
    plt = None
    sns = None

# --------------------------------------------------------------------------
# Instance-level constants, populated by bind_instance().
# Left as None so an unbound call fails loudly rather than silently using
# stale values from a previous instance.
# --------------------------------------------------------------------------
C = None                 # list of customer ids, 1..n
traveling_time = None    # {(i, j): euclidean distance}
customer_loads = None    # {node_id: demand}  (dict form; array form is passed explicitly)
time_windows = None      # {node_id: (release_time, deadline)}
release_time = None      # {node_id: release time}
eta = None               # energy consumption per unit distance
recharging_rate = None   # kW per unit time
f_recharging = None      # fixed recharging setup time


def bind_instance(*, C, traveling_time, customer_loads, time_windows,
                  release_time, eta, recharging_rate, f_recharging):
    """Bind the instance constants that the operator functions read as globals."""
    g = globals()
    g["C"] = C
    g["traveling_time"] = traveling_time
    g["customer_loads"] = customer_loads
    g["time_windows"] = time_windows
    g["release_time"] = release_time
    g["eta"] = eta
    g["recharging_rate"] = recharging_rate
    g["f_recharging"] = f_recharging


def instance_is_bound() -> bool:
    """True when bind_instance() has been called."""
    return C is not None and traveling_time is not None


# ==========================================================================
# Everything below is copied verbatim from main_ALNS.ipynb cell 1.
# ==========================================================================


def trip_distance(trip):
    d = 0.0
    for a, b in zip(trip[:-1], trip[1:]):
        d += traveling_time[(a, b)]
    return d


def check_capacity_g(trip, Q_v):
    load = 0.0
    for node in trip[1:-1]:
        load += customer_loads[node]
    return load <= Q_v


def objective_function_global(solution, pen_val, reward_val):
    cost = 0
    reven = 0
    penalty = 0
    Not_served = set(C)
    for i,j in solution.items():
        cost += trip_distance(j)
        for jj in j:
            Not_served.discard(jj)
    for i in Not_served:
        penalty += pen_val[i]
    Sr = set(C) - Not_served
    for i in Sr:
        reven += reward_val[i]
    return - reven + cost + penalty


def objective_function_value(sol, traveling_time, route_restr,
                             customer_loads, Q_v, b_v, print_status = False):
    #If there are no trips: return feasible with 0 cost, and save an empty schedule
    """
    Calculate the objective function value of a solution

    Parameters:
    - sol (list of routes/trips): list of sequence of trips.
    - traveling_time (dictionary):  {(from, to): time}.
    - time_windows (list of tuples): [(earliest time, latest arrival)].
    - b_v (float): battery capacity of selected vehicle
    - route_restr (int): maximum route length
    - customer_loads (np array): demand load of each customer starting at 0,..., n-1. You need to update index
    - Q_v (int): capacity of vehicle
    - recharging_rate (float): rate of charging in kW per unit of time.
    - f_recharging (float): fixed time for recharging setup.
    - release_time (dictionary): {node_id: release time} release time of each order.
    Returns:
    - Float: routing cost as total distance
    - Boolean: if vehicle routes satisfy time window, vehicle load capacity, and recharging plan restrictions
    """
    if sol is None or len(sol) == 0:
        objective_function_value.last_schedule = {"trips": [], "total_cost": 0.0, "feasible": True}
        return 0.0, True


    def trip_energy_needed(trip):
        return trip_distance(trip) * eta

    def trip_release_ready_time(trip):
        r = 0.0
        for node in trip:
            if node == 0:
                continue
            r = max(r, release_time.get(node, 0.0))
        return r

    def check_capacity(trip):
        load = 0.0
        for node in trip[1:-1]:
            load += customer_loads[node-1]
        return load <= Q_v

    def check_route_len(trip):
        return trip_distance(trip) <= route_restr

    #Simulate a trip from a specific depart time to get arrivals and slack
    def simulate_from_depart(depart_time, trip):
        t = depart_time
        arrivals = []
        min_slack = float("inf")
        for a, b in zip(trip[:-1], trip[1:]):
            t += traveling_time[(a, b)]
            (tw_e, tw_l) = time_windows[b]
            if t < tw_e:
                t = tw_e
            # wait until earliest window
            if t > tw_l:
                return None, None, False, 0.0
            arrivals.append((b, t))
            min_slack = min(min_slack, tw_l - t)
        return arrivals, t, True, (0.0 if min_slack == float("inf") else min_slack)

    total_cost = 0.0
    #schedule of the plan
    schedule = {"trips": [], "total_cost": 0.0, "feasible": False}

    clock = 0.0
    battery = float(b_v) if b_v is not None else float("inf")
    finite_battery = np.isfinite(b_v)

    for trip_idx, trip in enumerate(sol):
        if not trip or len(trip) < 2:
            continue

        # Standard feasibility checks
        if not check_capacity(trip):
            objective_function_value.last_schedule = {"trips": [], "total_cost": float("inf"), "feasible": False}
            print('capacity violation')
            return float("inf"), False

        if not check_route_len(trip):
            objective_function_value.last_schedule = {"trips": [], "total_cost": float("inf"), "feasible": False}
            return float("inf"), False

        trip_ready = trip_release_ready_time(trip)
        e_need = trip_energy_needed(trip)

        # Earliest possible depart (current time + release constraints)
        D0 = max(clock, trip_ready)

        #Dry-run from D0 just to measure “extra_delay_max”
        _, _, ok0, slack_after_depart = simulate_from_depart(D0, trip)
        if not ok0:
            objective_function_value.last_schedule = {"trips": [], "total_cost": float("inf"), "feasible": False}
            return float("inf"), False

        # Pre-wait available due to release time
        pre_wait = max(0.0, trip_ready - clock)
        #extra_delay_max from time-window slack
        extra_delay_max = slack_after_depart
        recharge_time_used = 0.0

        #Pre-departure charging plan (pre-wait + allowed departure delay)
        if finite_battery:
            deficit = max(0.0, e_need - battery)
            if deficit > 1e-9:
                if recharging_rate <= 0.0:
                    objective_function_value.last_schedule = {"trips": [], "total_cost": float("inf"), "feasible": False}
                    return float("inf"), False

                t_needed = f_recharging + deficit / recharging_rate
                available_time = pre_wait + extra_delay_max

                #Infeasible ONLY if charging cannot fit inside pre-wait + slack
                # fma, why not using available_time?
                if t_needed > available_time + 1e-9:
                    objective_function_value.last_schedule = {"trips": [], "total_cost": float("inf"), "feasible": False}
                    return float("inf"), False

                # Use pre-wait first, then delay depart by the remainder
                depart_delay = max(0.0, t_needed - pre_wait)
                clock = D0 + depart_delay

                # Only time after setup actually adds energy
                battery = min(b_v, battery + max(0.0, t_needed - f_recharging) * recharging_rate)
                recharge_time_used += t_needed
            else:
                clock = D0
        else:
            clock = D0

        #Final depart time (after any charging delay)
        depart_time = clock
        batt_start = battery

        #Simulate actual arrivals (used to report per-customer arrival times)
        arrivals, end_time, ok, _ = simulate_from_depart(depart_time, trip)
        if not ok:
            objective_function_value.last_schedule = {"trips": [], "total_cost": float("inf"), "feasible": False}
            return float("inf"), False

        # Consume energy for this trip; guard against negatives
        if finite_battery:
            battery = batt_start - e_need
            if battery < -1e-9:
                objective_function_value.last_schedule = {"trips": [], "total_cost": float("inf"), "feasible": False}
                return float("inf"), False
            if -1e-9 <= battery < 0.0:
                battery = 0.0

        this_trip_len = trip_distance(trip)
        total_cost += this_trip_len

        #Log full trip info: depart, recharge_before, battery levels, arrivals, end time
        schedule["trips"].append({
            "trip_index": trip_idx + 1,
            "depart_time": depart_time,
            "recharge_time_before": recharge_time_used,
            "battery_start": batt_start if finite_battery else float("inf"),
            "battery_end": battery if finite_battery else float("inf"),
            "arrivals": arrivals,        #per-customer arrival times
            "end_time": end_time,        #trip end time
            "trip_length": this_trip_len
        })

        # Next trip starts after this one ends
        clock = end_time

    #Save the overall result and return
    schedule["total_cost"] = total_cost
    schedule["feasible"] = True
    objective_function_value.last_schedule = schedule
    if print_status:
        print(schedule)
    return total_cost, True


def insert_node_in_all_positions(test, node, veh_cap, customer_loads):
    """
    Generates all possible insert positions of a node for a given vehicle existing routes.

    Parameters:
    - test: list of lists of the vehicle existing routes
    - node (int): node to be inserted
    - veh_cap (int): capacity of vehicle, pick the highest capacity
    - customer_loads (numpy): array of demand of each customer
    Returns:
    - dictionary of list of lists: solution of inserting the node at a position.
    """
    result = {}
    index = 1  # Dictionary index starts from 1
    sub_list = [0,node,0] # assuming a new trip to be added
    for list_idx in range(len(test)):  # Iterate over all sublists
        for pos in range(1, len(test[list_idx])):  # Avoid inserting before the first 0
            new_test = [lst[:] for lst in test]  # Deep copy to avoid modifying original list
            new_test[list_idx].insert(pos, node)  # Insert node at the given position
            result[index] = new_test
            index += 1
    for j in range(len(test)+1):
        dummy_test = [lst[:] for lst in test]
        dummy_test.insert(j, sub_list)
        result[index] = dummy_test
        index += 1
    remove_keys = []
    # remove infeasible routes due to capacity of vehicle violation
    for yt, item in result.items():
        condition = True
        for tm in item:
            ids = np.array(tm[1:-1]) - 1
            if np.sum([customer_loads[id] for id in ids]) > veh_cap:
                condition = False
        if not condition:
            remove_keys.append(yt)
    for ytt in remove_keys:
        del result[ytt]
    return result


def insert_node_in_all_positions_claude(test, node, veh_cap, customer_loads):
    """
    Generates all possible insert positions of a node for a given vehicle existing routes.

    Parameters:
    - test: list of lists of the vehicle existing routes
    - node (int): node to be inserted
    - veh_cap (int): capacity of vehicle, pick the highest capacity
    - customer_loads (numpy): array of demand of each customer
    Returns:
    - dictionary of list of lists: solution of inserting the node at a position.
    """
    result = {}
    index = 1
    node_demand = customer_loads[node - 1]  # Pre-fetch node demand once

    # Pre-compute current load for each trip in test
    trip_loads = []
    for trip in test:
        if len(trip) > 2:  # Has customers (not just [0, 0])
            trip_load = np.sum(customer_loads[np.array(trip[1:-1]) - 1])
        else:
            trip_load = 0
        trip_loads.append(trip_load)

    # Insert into existing trips
    for list_idx, trip in enumerate(test):
        current_load = trip_loads[list_idx]

        # Check if adding node would exceed capacity
        if current_load + node_demand > veh_cap:
            continue  # Skip this trip entirely - all positions infeasible

        # All positions in this trip are feasible, add them
        for pos in range(1, len(trip)):
            new_test = [lst[:] for lst in test]
            new_test[list_idx].insert(pos, node)
            result[index] = new_test
            index += 1

    # Insert as new trip [0, node, 0]
    if node_demand <= veh_cap:  # Only add if feasible
        sub_list = [0, node, 0]
        for j in range(len(test) + 1):
            new_test = [lst[:] for lst in test]
            new_test.insert(j, sub_list)
            result[index] = new_test
            index += 1

    return result


def route_length(sol, traveling_time):
    """
    Calculate the traveled distance of a given route

    Parameters:
    - sol (list): list of sequence of trips.
    - traveling_time (dictionary):  {(from, to): time}.
    Returns:
    - Float: route length as total distance
    """
    cost = 0
    for j in range(len(sol)-1):
        cost += traveling_time[(sol[j], sol[j+1])]
    return cost


def greedy_insert_claude(solution_dict, customer_id, customer_loads, traveling_time, Q,
                  b_v, max_route_length=np.inf):
    """
    Insert a node in the best position if feasible.

    Parameters:
    - solution_dict (dictionary): {(veh_id, trip_id): route[list]}
    - customer_id (int): node to be inserted.
    - customer_loads (numpy array): [demand of each customer].
    - traveling_time (dictionary): {(from, to): time}.
    - Q (list): load capacity of vehicles
    - b_v: battery capacity

    Returns:
    - solution_dict (dictionary): {(veh_id, trip_id): route[list]}
    """

    # ========================================================================
    # STEP 1: Pre-organize routes by vehicle (do once, not in loop)
    # ========================================================================
    routes_by_vehicle = {i: [] for i in range(1, len(Q) + 1)}
    for (veh_id, trip_id), route in solution_dict.items():
        routes_by_vehicle[veh_id].append(route)

    # ========================================================================
    # STEP 2: Pre-calculate cost for each vehicle (do once, not per insertion)
    # ========================================================================
    vehicle_costs = {}
    vehicle_feasible = {}

    for veh_id in range(1, len(Q) + 1):
        routes_list = routes_by_vehicle[veh_id]
        cost, feasible = objective_function_value(
            routes_list, traveling_time, max_route_length,
            customer_loads, Q[veh_id - 1], b_v
        )
        vehicle_costs[veh_id] = cost
        vehicle_feasible[veh_id] = feasible

        # Early exit if any vehicle is already infeasible
        if not feasible:
            return solution_dict

    # Calculate baseline total cost (all vehicles)
    baseline_cost = sum(vehicle_costs.values())

    # ========================================================================
    # STEP 3: Find best insertion across all vehicles
    # ========================================================================
    best_veh = None
    best_answer = None
    best_cost = np.inf

    for veh_id in range(1, len(Q) + 1):
        routes_list = routes_by_vehicle[veh_id]

        # Get all possible insertions for this vehicle
        assess_all = insert_node_in_all_positions_claude(
            routes_list, customer_id, Q[veh_id - 1], customer_loads
        )

        # Skip if no feasible insertions
        if not assess_all:
            continue

        # Cost of all OTHER vehicles
        other_vehicles_cost = baseline_cost - vehicle_costs[veh_id]

        # Evaluate each insertion option
        for key1, sample_route in assess_all.items():
            new_cost, feasible = objective_function_value(
                sample_route, traveling_time, max_route_length,
                customer_loads, Q[veh_id - 1], b_v
            )

            if not feasible:
                continue

            total_cost = new_cost + other_vehicles_cost

            if total_cost < best_cost:
                best_veh = veh_id
                best_cost = total_cost
                best_answer = {}
                for num, trip in enumerate(sample_route):
                    best_answer[(veh_id, num + 1)] = trip

    # ========================================================================
    # STEP 4: Apply best insertion if found
    # ========================================================================
    if best_answer:
        # Remove all trips from selected vehicle
        keys_to_delete = [k for k in solution_dict.keys() if k[0] == best_veh]
        for k in keys_to_delete:
            del solution_dict[k]

        # Add new trips
        solution_dict.update(best_answer)

    return solution_dict


def greedy_insert(solution_dict, customer_id,customer_loads,traveling_time,Q,
                  b_v, max_route_length = np.inf):
    """
    Insert a node in the best position if feasible.

    Parameters:
    - solution_dict (dictionary): {(veh_id, trip_id): route[list]} # key is tuple of veh_id, trip_id; value: route starting at the depot and ending at the depot.
    - customer_id (int):  node to be inserted.
    - customer_loads (numpy array): [demand of each customer].
    - traveling_time (dictionary):  {(from, to): time}.
    - Q (list): load capacity of vehicles
    - time_windows (list of tuples): [(earliest time, latest arrival)].

    Returns:
    - solution_dict (dictionary): {(veh_id, trip_id): route[list]}
    """
    # given a partially destroyed solution and a customer, insert the customer
    # at the best position respecting all constraints
    best_veh = None
    best_answer = None
    best_cost = np.inf
    # new_list = [list1[i] for i in indexx]
    for i in range(1, len(Q) + 1):
        routes_list = []
        # aggregate the routes of each vehicle
        for key, val in solution_dict.items():
            if key[0] == i:
                routes_list.append(val)
        # now routes is a dictionary for only "each" vehicle of its trips
        # routes_list (list of lists) [[route1], [route2], [route3] ]
        assess_all = insert_node_in_all_positions_claude(routes_list, customer_id, Q[i-1], customer_loads)
        # calculate the objective function value of each solution
        other_cost = 0 # calculate the cost of routing of all other vehicles not being assessed.
        for i2 in range(1, len(Q) + 1):
            if i2 != i:
                routes_list2 = []
                # aggregate the routes of each vehicle
                for key, val in solution_dict.items():
                    if key[0] == i2:
                        routes_list2.append(val)
                val, log = objective_function_value(routes_list2, traveling_time, max_route_length,
                                 customer_loads, Q[i2-1], b_v)
                other_cost += val
                if not log:
                    return solution_dict


        for key1, sample_route in assess_all.items():
            # sample_route [list of lists]: all trips of vehicle
            new_cost, feasible = objective_function_value(sample_route, traveling_time, max_route_length,
                             customer_loads, Q[i-1], b_v)
            if feasible and new_cost + other_cost < best_cost:
                best_veh = i
                best_cost = new_cost + other_cost
                best_answer = {}
                for num, trip in enumerate(sample_route):
                    best_answer[(best_veh, num + 1)] = trip # create a dictionary of (veh_id, trip_id): [route], namely all routes of the same vehicle regardless
    if best_answer:
        # remove all trips from selected vehicle since it will be inserted.
        keys_to_delete = [kyyy for kyyy, _ in solution_dict.items() if kyyy[0] == best_veh]
        for kyyy in keys_to_delete:
            del solution_dict[kyyy]

        solution_dict.update(best_answer)
        return solution_dict
    else:
        return solution_dict


def smallest_trips(solution_dict, traveling_time, num_trips):
    # Min-heap to store (route_length, veh_id, trip_id)
    heap = []
    # Populate the heap
    for (veh_id, trip_id), route in solution_dict.items():
        cost = route_length(route, traveling_time)
        heapq.heappush(heap, (cost, veh_id, trip_id))

    # Get the three shortest trips
    shortest_trip_keys = [(veh_id, trip_id) for _, veh_id, trip_id in heapq.nsmallest(num_trips, heap)]
    # Get all routes as a single flattened list
    flattened_routes = list(chain.from_iterable(solution_dict[key] for key in shortest_trip_keys))
    filtered_routes = [x for x in flattened_routes if x != 0]
    return filtered_routes


def largest_trips(solution_dict, traveling_time, num_trips):
    # Min-heap to store (route_length, veh_id, trip_id)
    heap = []
    # Populate the heap
    for (veh_id, trip_id), route in solution_dict.items():
        cost = route_length(route, traveling_time)
        heapq.heappush(heap, (cost, veh_id, trip_id))

    # Get the three shortest trips
    shortest_trip_keys = [(veh_id, trip_id) for _, veh_id, trip_id in heapq.nlargest(num_trips, heap)]
    # Get all routes as a single flattened list
    flattened_routes = list(chain.from_iterable(solution_dict[key] for key in shortest_trip_keys))
    filtered_routes = [x for x in flattened_routes if x != 0]

    return filtered_routes


def selected_trip(solution_dict, trip_id):
    '''
    trip_id (tuple): (veh_id, trip_id)
    '''
    shortest_trip_keys = [trip_id]
    # Get all routes as a single flattened list
    flattened_routes = list(chain.from_iterable(solution_dict[key] for key in shortest_trip_keys))
    filtered_routes = [x for x in flattened_routes if x != 0]

    return filtered_routes


def random_trips(solution_dict, num_trips):
    # random.sample raises when the solution holds fewer trips than requested,
    # including the empty case reachable under a tight battery when repair
    # cannot reinsert anything. Clamping is a no-op whenever the original
    # worked, since min() only binds once len(keys) < num_trips.
    keys = list(solution_dict.keys())
    if not keys:
        return []
    shortest_trip_keys = random.sample(keys, min(num_trips, len(keys)))
    # Get all routes as a single flattened list
    flattened_routes = list(chain.from_iterable(solution_dict[key] for key in shortest_trip_keys))
    filtered_routes = [x for x in flattened_routes if x != 0]

    return filtered_routes


def zone_removal(radius, locations):
    # Remove the depot or first row (based on node_id)
    new_locations = locations[locations["node_id"] != 0]

    # List of node_id values available to sample from
    candidate_ids = new_locations["node_id"].tolist()

    # Randomly select 2 node_id values
    selected_node_ids = random.sample(candidate_ids, 2)

    # Get X,Y for the selected nodes
    x_selected = new_locations.loc[new_locations["node_id"].isin(selected_node_ids), "X"].values
    y_selected = new_locations.loc[new_locations["node_id"].isin(selected_node_ids), "Y"].values

    # Compute distances to both selected nodes
    distances = np.sqrt((new_locations["X"] - x_selected) ** 2 +
                        (new_locations["Y"] - y_selected) ** 2)

    # Get node_id values of points within the radius
    remove_list = new_locations.loc[distances <= radius, "node_id"].tolist()

    return remove_list


def selected_zone_removal(radius, locations, coordinate_tuple):
    new_locations = locations.copy()
    new_locations = new_locations.iloc[1:,:]
    new_locations['distance'] = np.sqrt((new_locations['X'] - coordinate_tuple[0]) ** 2 + (new_locations['Y'] - coordinate_tuple[1]) ** 2)
    removal = new_locations[new_locations['distance'] <= radius][['X', 'Y']].index.tolist()

    return removal


def worst_distance_nodes(current_sol, traveling_time, num_remove=5):
    """
    Identify and remove the top worst nodes based on the distance score.
    Parameters:
    - current_sol: dictionary {(veh_id, trip_id): route}
    - traveling_time: dictionary {(from, to): distance}
    - num_remove: Number of nodes to remove (default = 5)

    Returns:
    - worst_nodes: List of removed node IDs
    """
    heap = []  # Max-heap to store scores (-score, node_id)
    routes = []
    for _, i in current_sol.items():
        routes.append(i)
    # Compute scores for each node in all routes
    for route in routes:
        for i in range(1, len(route) - 1):  # Skip first and last nodes
            j, k = route[i - 1], route[i + 1]  # Previous and next nodes
            i_node = route[i]

            # Compute score
            score = traveling_time[(j, i_node)] + traveling_time[(i_node, k)] - traveling_time[(j, k)]

            # Push into heap (negative score for max-heap behavior)
            heapq.heappush(heap, (-score, i_node))

    # Extract the top 'num_remove' highest-scoring nodes
    worst_nodes = [heapq.heappop(heap)[1] for _ in range(min(num_remove, len(heap)))]

    return worst_nodes


def remove_nodes_from_routes(routes, nodes_to_remove):
    """
    Creates a copy of the given routes dictionary and removes the specified nodes from all routes.

    Parameters:
    - routes (dict): Dictionary where keys are (veh_id, trip_id) and values are lists of nodes.
    - nodes_to_remove (set): Set of nodes to be removed.

    Returns:
    - dict: A new dictionary with the nodes removed.
    """
    # Make a deep copy of the dictionary
    new_routes = copy.deepcopy(routes)

    # Remove nodes from each route
    for key in new_routes:
        new_routes[key] = [node for node in new_routes[key] if node not in nodes_to_remove]

    return new_routes


def seq_removal(current_sol, num_nodes):
    routes = list(current_sol.values())
    z = []
    for i in routes:
        z.extend(i)
    z = [x for x in z if x!=0]
    location = len(z)
    half = int(num_nodes/2)
    # The original called random.randint(half, location), which raises when the
    # incumbent serves fewer than `half` customers -- reachable under a tight
    # battery after a destroy step whose repair finds no feasible insertion.
    # At location == half the original returns the whole of z, so returning z
    # for location <= half extends that behaviour without changing any case
    # that previously worked.
    if location <= half:
        return list(z)
    some_where = random.randint(half, location)
    return z[some_where - half: min(len(z), some_where + half + 1)]


def gen_coordinate_list(x_dim_max, y_dim_max,x_dim_min, y_dim_min, num_zones, min_distance):
    """
    x_dim_max, y_dim_max,x_dim_min, y_dim_min (float): corners of the map
    num_zones (int): how many points to generate
    min_distance (float): value of minimum distance between any two points
    return coordinat (list): [(x1, y1), (x2,y2),...,(x_n,y_n)] N locations that are far from each other at least by min_distance
    """
    coordinat = []
    while(True):
        new_loc = (random.uniform(x_dim_min, x_dim_max),random.uniform(y_dim_min, y_dim_max))
        coordinat.append(new_loc)
        if len(coordinat) >= 2:
            for i in range(len(coordinat)-1):
                for j in range(i+1, len(coordinat)):
                    distance = ((coordinat[i][0]-coordinat[j][0])**2+
                    (coordinat[i][1]-coordinat[j][1])**2)**0.5
                    if distance <= min_distance:
                        coordinat.pop(j)
                        coordinat.pop(i)
                        continue

        if len(coordinat) == num_zones:
            break
    return coordinat


def plot_vrp_multitrip(routing, locations_df, new_set, end_unserved, global_best_val_alns):
    """
    routing: dict with keys (veh_id, trip_id) → list of nodes visited in order
    locations_df: contains columns ['node_id','X','Y']
    new_set: set/list of served nodes
    end_unserved: set/list of unserved nodes
    global_best_val_alns: numeric objective function value
    """

    # Prepare lookup for node coordinates
    node_to_xy = locations_df.set_index("node_id")[["X", "Y"]].to_dict("index")

    # Colors for vehicles
    vehicle_colors = {}
    color_map = plt.cm.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(12, 10))

    # -----------------------------
    # Draw Routes
    # -----------------------------
    for (veh_id, trip_id), route in routing.items():
        if veh_id not in vehicle_colors:
            vehicle_colors[veh_id] = color_map(len(vehicle_colors) % 10)

        color = vehicle_colors[veh_id]

        # get coordinates for the route
        xs = [node_to_xy[n]["X"] for n in route]
        ys = [node_to_xy[n]["Y"] for n in route]

        # Draw polyline for this trip
        ax.plot(xs, ys, "-", linewidth=2, color=color,
                label=f"Vehicle {veh_id} Trip {trip_id}")

        # Draw nodes in the route
        ax.scatter(xs, ys, s=60, color=color)

    # -----------------------------
    # Draw Served Nodes
    # -----------------------------
    served_coords = locations_df[locations_df["node_id"].isin(new_set)]
    ax.scatter(
        served_coords["X"], served_coords["Y"],
        marker="o", color="green", s=80, label="Served Nodes"
    )

    # -----------------------------
    # Draw Unserved Nodes
    # -----------------------------
    unserved_coords = locations_df[locations_df["node_id"].isin(end_unserved)]

    ax.scatter(
        unserved_coords["X"], unserved_coords["Y"],
        marker="X", color="red", s=120, label="Unserved Nodes"
    )

    # Add labels next to each unserved node
    for _, row in unserved_coords.iterrows():
        ax.text(
            row["X"] + 0.5,      # slight offset so label is readable
            row["Y"] + 0.5,
            str(int(row["node_id"])),
            fontsize=10,
            color="red",
            weight="bold"
        )

    # -----------------------------
    # Depot Highlight (node 0)
    # -----------------------------
    depot_x = node_to_xy[0]["X"]
    depot_y = node_to_xy[0]["Y"]
    ax.scatter([depot_x], [depot_y], color="blue", s=200, marker="s", label="Depot")

    # -----------------------------
    # Write Objective Function Value
    # -----------------------------
    ax.text(
        0.98, 0.02,
        f"Objective Value: {global_best_val_alns:.2f}",
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7)
    )

    ax.set_title("Vehicle Routing Problem with Multiple Trips", fontsize=16)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def generate_non_repeating_integers(start, end, count):
    """
    Generates a list of random, non-repeating integers within a specified range.

    Args:
        start (int): The inclusive start of the range.
        end (int): The inclusive end of the range.
        count (int): The number of unique integers to generate.

    Returns:
        list: A list of 'count' random, non-repeating integers.
    """
    if count > (end - start + 1):
        raise ValueError("Count cannot be greater than the range size")

    # Create the population range and sample from it
    population = range(start, end + 1)
    return random.sample(population, count)