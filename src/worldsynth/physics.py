from __future__ import annotations

from pxr import Gf, UsdPhysics

from .scene import set_transform, world_transform


class IsaacLabSettler:

    def __init__(self, steps: int):
        if steps < 1:
            raise ValueError(f"physics_steps must be at least 1, got {steps}")
        from isaaclab.sim import SimulationCfg, SimulationContext

        self.steps = steps
        self.sim = SimulationContext(SimulationCfg(device="cpu", dt=1.0 / 120.0, use_fabric=False))

    def settle(self, scene, clip: dict, objects: list):
        bodies = []
        for part, obj in zip(clip["parts"], objects, strict=True):
            prim = scene.stage.GetPrimAtPath(obj.prim_path)
            body = UsdPhysics.RigidBodyAPI(prim)
            if not body:
                raise RuntimeError(f"Part {obj.prim_path} has no rigid-body API")
            
            collider_prim = prim.GetChild("physics_collision")
            collision = UsdPhysics.CollisionAPI(collider_prim)
            if not collision:
                raise RuntimeError(f"Part {obj.prim_path} has no collision API")
            
            # Apply MeshCollisionAPI and set the approximation to convex hull
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider_prim)
            mesh_collision.GetApproximationAttr().Set("convexHull")

            collision.GetCollisionEnabledAttr().Set(False)
            body.GetKinematicEnabledAttr().Set(True)
            body.GetVelocityAttr().Set(Gf.Vec3f(0.0))
            body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0))
            bodies.append((prim, body, collision, part, obj))

        self.sim.reset()
        for prim, body, collision, part, obj in bodies:
            collision.GetCollisionEnabledAttr().Set(True)
            body.GetKinematicEnabledAttr().Set(False)
            for _ in range(self.steps):
                self.sim.step(render=False)
            settled = world_transform(prim)
            body.GetVelocityAttr().Set(Gf.Vec3f(0.0))
            body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0))
            body.GetKinematicEnabledAttr().Set(True)
            part["pose"] = settled
            obj.T_world_obj = settled

        self.sim.stop()
        for prim, _, _, part, _ in bodies:
            set_transform(prim, part["pose"])