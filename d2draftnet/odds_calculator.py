def calculate_site_margin(odds_team1, odds_team2):
    """
    Calculate the betting site's margin and implied 50/50 penalty.
    
    Args:
        odds_team1 (float): Odds for Team 1.
        odds_team2 (float): Odds for Team 2.
        
    Returns:
        dict: Margin and penalty details.
    """
    # Calculate implied probabilities
    p_implied_1 = 1 / odds_team1
    p_implied_2 = 1 / odds_team2

    # Total implied probability and margin
    total_implied_prob = p_implied_1 + p_implied_2
    margin = total_implied_prob - 1
    margin_per_team = margin / 2  # Approximate margin per team

    # Return results as a dictionary
    return {
        "implied_prob_team1": p_implied_1,
        "implied_prob_team2": p_implied_2,
        "total_implied_prob": total_implied_prob,
        "margin": margin,
        "margin_per_team": margin_per_team,
    }


# Example Usage
if __name__ == "__main__":
    # Example odds
    odds_team1 = 1.2
    odds_team2 = 4.2

    # Call the function and display results
    results = calculate_site_margin(odds_team1, odds_team2)

    # Optionally, use returned data
    print("\nReturned Results:")
    for key, value in results.items():
        print(f"{key}: {value:.2%}" if isinstance(value, float) else f"{key}: {value}")
