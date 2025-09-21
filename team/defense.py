class DefensiveSkill:
    @staticmethod
    def validate_static(
            blitzing: float=0.5,
            rush_defense: float=0.5,
            pass_defense: float=0.5,
            coverage: float=0.5,
            turnovers: float=0.5,
            penalties: float=0.5
        ) -> tuple[bool, str]:
        """
        Validates the DefensiveSkill properties

        Args:
            blitzing (float): How good the defense is at blitzing
            rush_defense (float): How good the defense is at rush defense
            pass_defense (float): How good the defense is at pass defense
            coverage (float): How good the defense is at coverage
            turnovers (float): How likely the defense is to force a turnover
            penalties (float): How unlikely the defense is to commit a penalty
        
        Returns:
            bool: Whether the properties are valid
            str: Error message if the properties are invalid
        """
        if blitzing > 1.0 or blitzing < 0:
            return False, f"Blitzing out of bounds (0 - 1): {blitzing}"
        if rush_defense > 1.0 or rush_defense < 0:
            return False, f"Rush defense out of bounds (0 - 1): {rush_defense}"
        if pass_defense > 1.0 or pass_defense < 0:
            return False, f"Pass defense out of bounds (0 - 1): {pass_defense}"
        if coverage > 1.0 or coverage < 0:
            return False, f"Coverage out of bounds (0 - 1): {coverage}"
        if turnovers > 1.0 or turnovers < 0:
            return False, f"Turnovers out of bounds (0 - 1): {turnovers}"
        if penalties > 1.0 or penalties < 0:
            return False, f"Penalties out of bounds (0 - 1): {penalties}"
        return True, ""
    
    def __init__(
            self,
            blitzing: float=0.5,
            rush_defense: float=0.5,
            pass_defense: float=0.5,
            coverage: float=0.5,
            turnovers: float=0.5,
            penalties: float=0.5
        ) -> "DefensiveSkill":
        """
        Constructor for the DefensiveSkill class

        Args:
            blitzing (float): How good the defense is at blitzing
            rush_defense (float): How good the defense is at rush defense
            pass_defense (float): How good the defense is at pass defense
            coverage (float): How good the defense is at coverage
            turnovers (float): How likely the defense is to force a turnover
            penalties (float): How unlikely the defense is to commit a penalty
        
        Returns:
            DefensiveSkill: The instantiated DefensiveSkill object
        """
        # Validate the defensive skill properties
        valid, err = DefensiveSkill.validate_static(
            blitzing=blitzing,
            rush_defense=rush_defense,
            pass_defense=pass_defense,
            coverage=coverage,
            turnovers=turnovers,
            penalties=penalties
        )
        if not valid:
            raise ValueError(err)
        
        # Save as object properties
        self.blitzing = blitzing
        self.rush_defense = rush_defense
        self.pass_defense = pass_defense
        self.coverage = coverage
        self.turnovers = turnovers
        self.penalties = penalties
