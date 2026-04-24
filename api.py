from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from spiral_torsion_spring_optimizer import SpiralTorsionSpring

app = FastAPI()

class MaximizeStiffnessRequest(BaseModel):
    elasticity: float = Field(gt=0)
    stress_yield: float = Field(gt=0)
    height: float = Field(gt=0)
    max_radius_pre: float = Field(gt=0)
    radius_center: float = Field(ge=0)
    pitch_0: float = Field(gt=0)
    deltatheta_opt: float = Field(gt=0)
    torque_pre: float = Field(ge=0)
    safety_factor: float = Field(gt=0)
    max_thickness: float | None = Field(default=None, gt=0)
    nozzle_diameter: float = Field(gt=0)
    opt_params: dict | None = Field(default=None)

class NoFeasibleSolutionError(Exception):
    pass

@app.exception_handler(NoFeasibleSolutionError)
async def no_feasible_handler(request: Request, exc: NoFeasibleSolutionError):
    return JSONResponse(
        status_code=422,
        content={"error": "no_feasible_solution", "message": str(exc)},
    )

@app.post("/v1/maximize_stiffness")
def maximize_stiffness(req: MaximizeStiffnessRequest):
    try:
        req_data = req.model_dump()
        opt_params = req_data.pop('opt_params', None)
        spring = SpiralTorsionSpring.maximize_stiffness(
            req_data,
            opt_params=opt_params
        )
        if spring.res.success:
            return spring.to_dict()
        else:
            raise NoFeasibleSolutionError()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}