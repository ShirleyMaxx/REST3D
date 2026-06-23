import os


def generate_urdf(object_path, object_name, output_dir="./", prefix="scene_canon_"):
    template = '''<?xml version="1.0"?>
<robot name="{object_name}">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="{object_path}/{prefix}{object_name}.obj" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{object_path}/{prefix}{object_name}.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia
        ixx="0.1" ixy="0" ixz="0"
        iyy="0.1" iyz="0"
        izz="0.1"/>
    </inertial>
  </link>
</robot>'''
    
    content = template.format(object_path=object_path, object_name=object_name, prefix=prefix)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}{object_name}.urdf")
    with open(output_path, 'w') as f:
        f.write(content)


def generate_urdf_files(object_names, output_urdf_canon_dir, prefix="scene_canon_"):
    for object_name in object_names:
        generate_urdf('../obj_files', object_name, output_urdf_canon_dir, prefix=prefix)
