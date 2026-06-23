# Installation

REST3D uses two conda environments: `rest3d` for scene reconstruction (Stages 1–2) and `gym` for scene stabilization using Isaac Gym (Stage 3).

**1. Clone this repo**
```bash
git clone https://github.com/ShirleyMaxx/REST3D.git REST3D
cd REST3D
export REST3D_ROOT="$(pwd)"
```

**2. Set up the `rest3d` environment** — follow [environments/install_rest3d.md](environments/install_rest3d.md)

**3. Set up the `gym` environment** — follow [environments/install_gym.md](environments/install_gym.md)

**4. Set your API key** (for the `rest3d` environment):
```bash
# default Gemini backend
export GEMINI_API_KEY=...
```
