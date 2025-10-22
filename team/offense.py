class OffensiveSkill:
    @staticmethod
    def validate_static(
            blocking: float=0.5,
            rushing: float=0.5,
            passing: float=0.5,
            receiving: float=0.5,
            scrambling: float=0.5,
            turnovers: float=0.5,
            penalties: float=0.5,
            field_goals: float=0.5
        ) -> tuple[bool, str]:
        """
        Validates the OffensiveSkill properties

        Args:
            blocking (float): How good the offense is at blocking
            rushing (float): How good the offense is at rushing
            passing (float): How good the offense is at passing
            receiving (float): How good the offense is at receiving
            scrambling (float): How likely the quarterback is to scramble
            turnovers (float): How unlikely the offense is to commit a turnover
            penalties (float): How unlikely the offense is to commit a penalty
            field_goals (float): How good the offense is at kicking field goals
        
        Returns:
            bool: Whether the properties are valid
            str: Error message if the properties are invalid
        """
        if blocking > 1.0 or blocking < 0:
            return False, f"Blocking out of bounds (0 - 1): {blocking}"
        if rushing > 1.0 or rushing < 0:
            return False, f"Rushing out of bounds (0 - 1): {rushing}"
        if passing > 1.0 or passing < 0:
            return False, f"Passing out of bounds (0 - 1): {passing}"
        if receiving > 1.0 or receiving < 0:
            return False, f"Receiving out of bounds (0 - 1): {receiving}"
        if scrambling > 1.0 or scrambling < 0:
            return False, f"Scrambling out of bounds (0 - 1): {scrambling}"
        if turnovers > 1.0 or turnovers < 0:
            return False, f"Turnovers out of bounds (0 - 1): {turnovers}"
        if penalties > 1.0 or penalties < 0:
            return False, f"Penalties out of bounds (0 - 1): {penalties}"
        if field_goals > 1.0 or field_goals < 0:
            return False, f"Field goals out of bounds (0 - 1): {field_goals}"
        return True, ""
    
    def __init__(
            self,
            blocking: float=0.5,
            rushing: float=0.5,
            passing: float=0.5,
            receiving: float=0.5,
            scrambling: float=0.5,
            turnovers: float=0.5,
            penalties: float=0.5,
            field_goals: float=0.5
        ) -> "OffensiveSkill":
        """
        Constructor for the OffensiveSkill class

        Args:
            blocking (float): How good the offense is at blocking
            rushing (float): How good the offense is at rushing
            passing (float): How good the offense is at passing
            receiving (float): How good the offense is at receiving
            scrambling (float): How likely the quarterback is to scramble
            turnovers (float): How unlikely the offense is to commit a turnover
            penalties (float): How unlikely the offense is to commit a penalty
            field_goals (float): How good the offense is at kicking field goals
        
        Returns:
            OffensiveSkill: The instantiated OffensiveSkill object
        """
        # Validate the offensive skill properties
        valid, err = OffensiveSkill.validate_static(
            blocking=blocking,
            rushing=rushing,
            passing=passing,
            receiving=receiving,
            scrambling=scrambling,
            turnovers=turnovers,
            penalties=penalties,
            field_goals=field_goals
        )
        if not valid:
            raise ValueError(err)
        
        # Save as object properties
        self.blocking = blocking
        self.rushing = rushing
        self.passing = passing
        self.receiving = receiving
        self.scrambling = scrambling
        self.turnovers = turnovers
        self.penalties = penalties
        self.field_goals = field_goals
