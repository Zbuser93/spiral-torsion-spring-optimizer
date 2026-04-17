from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from SpiralTorsionSpringOptimizer import SpiralTorsionSpring

app = FastAPI()

class OptimizeRequest(BaseModel):
    input_height: float = Field(gt=0)
    input_elasticity: float = Field(gt=0)
    input_max_radius_pre: float = Field(gt=0)
    input_radius_center: float = Field(ge=0)
    input_pitch_0: float = Field(gt=0)
    input_deltatheta_opt: float = Field(gt=0)
    input_torque_pre: float = Field(ge=0)
    input_safety_factor: float = Field(gt=0)
    input_stress_yield: float = Field(gt=0)
    input_max_thickness: float | None = Field(default=None, gt=0)
    input_nozzle_diameter: float = Field(gt=0)
    seed: int | None = None

@app.post("/v1/optimize")
def optimize(req: OptimizeRequest):
    try:
        data = req.model_dump()
        seed = data.pop("seed", None)

        if seed is not None:
            import numpy as np
            np.random.seed(seed)

        spring = SpiralTorsionSpring.maximize_stiffness(data)
        return spring.to_dict()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}