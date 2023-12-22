import json
import random


def generate_synthetic_data(num_orders, num_drivers_per_order):
    synthetic_dataset = []

    for order_id in range(1, num_orders + 1):

        # ......................................................
        # {DistanceToDropOff} gets a random value to represent
        # the distance between the order's pickup location
        # and it's drop-off location.
        # The value is between 1 & 10 miles
        # ......................................................

        order = {
            "OrderID": order_id,
            "OrderType": 0,  # 0 for product, 1 for food
            "OrderPriority": 0,  # 0 for standard, 1 for express
            "DistanceToDropOff": round(random.uniform(1.0, 10.0), 2)
        }

        drivers = []

        order_distance_to_drop_off = order['DistanceToDropOff']

        # ......................................................
        # {current_orders} gets a random value to represent
        # the rider's current number of orders for delivery
        # including the new one about to be assigned to the rider
        # ......................................................

        for driver_id in range(1, num_drivers_per_order + 1):
            current_orders = random.randint(1, 3)

            # ......................................................
            # {online_status} gets the current status of the rider:
            # if he is currently moving or on standby
            #
            # Online status can be 0 for Standby or 1 for Moving
            # ......................................................

            online_status = random.randint(0, 1)

            # ......................................................
            # {acc_distance_from_here_to_pickup} gets the accumulated distance
            # from the rider's current location to the pickup location of the new
            # order, assuming the new order will be picked up last for delivery.
            #
            # Here, the acc_distance_from_here_to_pickup will be a random value
            # between 1 to 8 miles, plus the distance from the last order's
            # pickup location to it's drop-off location
            # ......................................................

            acc_distance_from_here_to_pickup = round(random.uniform(1.0, 8.0), 2)
            acc_distance_from_here_to_pickup = acc_distance_from_here_to_pickup + order_distance_to_drop_off

            # ......................................................
            # {total_acc_distance} will be used to get the rider's accumulated distance
            # accounting for all the rider's orders, plus a random factor to account
            # for varying distance between each order's pickup and drop-off locations
            # ......................................................

            total_acc_distance = (acc_distance_from_here_to_pickup * current_orders) + round(random.uniform(5.0, 15.0), 2)

            # ......................................................
            # if the rider is currently moving and not on standby,
            # then reduce the rider's accumulated distance by a factor
            # to introduce a bias that a rider on the move is more
            # likely to deliver a new item faster than one on Standby
            # ......................................................

            if online_status == 1:
                factor = round(random.uniform(3.0, 9.0), 2)
                total_acc_distance = total_acc_distance - factor

            total_acc_distance = round(total_acc_distance, 3)

            # ......................................................
            # {eta} will represent the total time it takes
            # for the rider to complete all his deliveries
            # with the new order included
            # ......................................................

            eta = total_acc_distance

            # ......................................................
            # {eta_factor} increases the delivery time to
            # account for time lost when picking up the order
            # from the pickup point, and time lost when
            # giving the item to the customer
            # ......................................................

            eta_factor = random.randint(8, 13)
            eta = round(eta + eta_factor)

            # print(f'Distance4: {eta}')

            driver = {
                "DriverID": driver_id,
                "AccDistanceFromHereToPickup": total_acc_distance,
                # "DistanceToNextDropOff": distance_to_next_drop_off,
                "ETA": eta,
                'OnlineStatus': online_status,
                'CurrentOrders': current_orders,
            }

            drivers.append(driver)

        synthetic_dataset.append({"order": order, "drivers": drivers})

    return {"synthetic_dataset": synthetic_dataset}


def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# Example usage:


orders = 100
drivers_per_order = 20

synthetic_data = generate_synthetic_data(orders, drivers_per_order)

# Print the generated synthetic data
print(synthetic_data)

# Save the data to a JSON file
save_to_json(synthetic_data, 'synthetic_data.json')
