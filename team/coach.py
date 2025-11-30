class CoachSkill:
    @staticmethod
    def validate_static(
            risk_taking: float=0.5,
            run_pass: float=0.5
        ) -> tuple[bool, str]:
        """
        Validates the CoachSkill properties

        Args:
            risk_taking (float): The coach's risk-taking tendency
            run_pass (float): The coach's run-pass playcalling tendency
        
        Returns:
            bool: Whether the properties are valid
            str: Error message if the properties are invalid
        """
        if risk_taking > 1.0 or risk_taking < 0:
            return False, f"Risk taking out of bounds (0 - 1): {risk_taking}"
        if run_pass > 1.0 or run_pass < 0:
            return False, f"Run-pass out of bounds (0 - 1): {run_pass}"
        return True, ""
    
    def __init__(
            self,
            risk_taking: float=0.5,
            run_pass: float=0.5
        ) -> "CoachSkill":
        """
        Constructor for the CoachSkill class

        Args:
            risk_taking (float): The coach's risk-taking tendency
            run_pass (float): The coach's run-pass playcalling tendency
        
        Returns:
            CoachSkill: The instantiated CoachSkill object
        """
        # Validate the defensive skill properties
        valid, err = CoachSkill.validate_static(
            risk_taking=risk_taking,
            run_pass=run_pass
        )
        if not valid:
            raise ValueError(err)
        
        # Save as object properties
        self.risk_taking = risk_taking
        self.run_pass = run_pass
