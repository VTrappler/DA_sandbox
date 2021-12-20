class DAModel:
    """
    A DAModel is defined as
    x_{t+1} = M_t(x_t) + w_t
    y_t = H_t(x_t) + v_k

    Cov[v_t] = R_t -> Observation Error
    Cov[w_t] = Q_k -> Model Error
    """

    def __init__(self, dim, Nintegration):
        "docstring"
        self.dim = dim
        self.integrator_steps = Nintegration

    def integrator_step(self, t, x, dt):
        pass

    def forward(self, t, x):
        for i in range(self.integrator_steps):
            self.integrator_step(t, x)
            pass

    def TLM(self, t, x):
        pass

    def observation_operator(self, t):
        """H"""
        pass
