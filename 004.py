import numpy as np

def calculate_free_energy(Q, complexity, prior):
    epsilon = 1e-10
    return -np.log(Q + epsilon) + complexity - np.log(prior + epsilon)

def update_belief(current_belief):
    # Ask the user for feedback
    outcome_happened = input("Did the desired outcome happen? (yes/no): ")
    if outcome_happened.lower() == 'yes':
        # Increase the belief (up to a maximum of 1)
        new_belief = min(current_belief + 0.1, 1.0)
    else:
        # Decrease the belief (down to a minimum of 0)
        new_belief = max(current_belief - 0.1, 0.0)
    
    # # Allow the user to manually adjust the confidence if desired
    # adjust_confidence = input("Would you like to manually adjust the confidence in this action? (yes/no): ")
    # if adjust_confidence.lower() == 'yes':
    #     new_belief = float(input("Enter the new confidence level (0 to 1): "))
    
    return new_belief

def decision_making_tool_multiple_steps():
    # Same initial steps as before
    problem = input("What problem do you want to solve? ")
    actions = input("List possible actions separated by commas: ").split(",")
    initial_beliefs = {}
    empirical_priors = {}
    complexities = {}
    for action in actions:
        belief = float(input(f"Initial belief that '{action.strip()}' will solve the problem (0 to 1): "))
        initial_beliefs[action] = belief
        prior = float(input(f"Initial preference for '{action.strip()}' (0 to 1): "))
        empirical_priors[action] = prior
        complexity = float(input(f"Complexity of taking action '{action.strip()}' (higher means more complex): "))
        complexities[action] = complexity
    
    # Loop for multiple timesteps
    while True:
        print("\nCalculating the best action to take...")
        free_energies = {}
        for action in actions:
            free_energy = calculate_free_energy(initial_beliefs[action], complexities[action], empirical_priors[action])
            free_energies[action] = free_energy
        
        recommended_action = min(free_energies, key=free_energies.get)
        print(f"The recommended action to solve '{problem}' is: {recommended_action.strip()}")
        
        # Update beliefs based on user feedback
        initial_beliefs[recommended_action] = update_belief(initial_beliefs[recommended_action])
        
        # Ask if the user wants to continue
        continue_decision = input("Would you like to continue? (yes/No): ")
        if continue_decision.lower() == 'no' or len(continue_decision.lower()) == 0:
            break

# Run the extended decision-making tool
# Note: To run this, you'd need to copy the code into your local Python environment
# decision_making_tool_multiple_steps()
if __name__ == "__main__":
    decision_making_tool_multiple_steps()