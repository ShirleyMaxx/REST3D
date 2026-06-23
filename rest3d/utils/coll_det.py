"""Batched GJK collision detection and geometric penetration rewards (torch)."""
import torch


def quat_to_rotmat_batch(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]
    R = torch.stack([
        1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w),
          2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w),
          2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y),
    ], dim=1).view(B, 3, 3)
    return R


def gjk_support(Va, Vb, d):
    ar = torch.arange(Va.shape[0], device=Va.device)
    ia = (Va *  d.unsqueeze(1)).sum(-1).argmax(1)
    ib = (Vb * (-d).unsqueeze(1)).sum(-1).argmax(1)
    return Va[ar, ia] - Vb[ar, ib]


def gjk_batch(Va, Vb, max_iter: int = 32):
    B, device = Va.shape[0], Va.device
    EPS = 1e-7

    def dot3(a, b):   return (a * b).sum(-1)
    def cross3(a, b):
        ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
        bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
        return torch.stack([ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx], -1)
    def unit(a):      return a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    def w3(m, a, b):  return torch.where(m.unsqueeze(1), a, b)
    def w1(m, a, b):  return torch.where(m, a, b)

    d = Va.mean(1) - Vb.mean(1)
    d = torch.where(d.norm(dim=-1, keepdim=True) < EPS,
                    d.new_tensor([[1., 0., 0.]]).expand(B, -1), d)
    d = unit(d)

    sx  = torch.zeros(B, 4, 3, device=device)
    cnt = torch.zeros(B, dtype=torch.long, device=device)
    sx[:, 0] = gjk_support(Va, Vb, d)
    cnt[:] = 1
    d = -sx[:, 0]

    hit    = torch.zeros(B, dtype=torch.bool, device=device)
    active = torch.ones(B,  dtype=torch.bool, device=device)

    for _ in range(max_iter):
        if not active.any():
            break
        zero_d = active & (d.norm(dim=-1) < EPS)
        hit    = hit | zero_d
        active = active & ~zero_d
        if not active.any():
            break
        A = gjk_support(Va, Vb, unit(d))
        no_pass = active & (dot3(A, unit(d)) < -EPS)
        active  = active & ~no_pass
        if not active.any():
            break
        for c in range(4):
            m = active & (cnt == c)
            sx[:, c] = w3(m, A, sx[:, c])
        cnt = w1(active, cnt + 1, cnt)

        m2 = active & (cnt == 2)
        if m2.any():
            Av2, Bv2 = sx[:, 1], sx[:, 0]
            AB = Bv2 - Av2
            AO = -Av2
            keep_AB = m2 & (dot3(AB, AO) > 0)
            keep_A  = m2 & ~keep_AB
            d_AB = cross3(cross3(AB, AO), AB)
            d_AB = torch.where((d_AB.norm(dim=-1) < EPS).unsqueeze(1), AO, d_AB)
            d = w3(keep_AB, d_AB, d)
            d = w3(keep_A, AO, d)
            sx[:, 0] = w3(keep_A, Av2, sx[:, 0])
            cnt = w1(keep_A, torch.ones_like(cnt), cnt)

        m3 = active & (cnt == 3)
        if m3.any():
            Cv3, Bv3, Av3 = sx[:, 0], sx[:, 1], sx[:, 2]
            AB = Bv3 - Av3
            AC = Cv3 - Av3
            AO = -Av3
            ABCn = cross3(AB, AC)
            nAC  = cross3(ABCn, AC)
            nAB  = cross3(AB, ABCn)
            reg_AC  = m3 & (dot3(nAC, AO) > 0)
            reg_AB  = m3 & ~reg_AC & (dot3(nAB, AO) > 0)
            reg_tri = m3 & ~reg_AC & ~reg_AB
            close_AC = reg_AC & (dot3(AC, AO) > 0)
            far_AC   = reg_AC & ~close_AC
            d_AC = cross3(cross3(AC, AO), AC)
            d_AC = torch.where((d_AC.norm(dim=-1) < EPS).unsqueeze(1), AO, d_AC)
            d = w3(close_AC, d_AC, d)
            d = w3(far_AC, AO, d)
            sx[:, 0] = w3(close_AC, Cv3, sx[:, 0])
            sx[:, 1] = w3(close_AC, Av3, sx[:, 1])
            cnt = w1(close_AC, torch.full_like(cnt, 2), cnt)
            sx[:, 0] = w3(far_AC, Av3, sx[:, 0])
            cnt = w1(far_AC, torch.ones_like(cnt), cnt)
            close_AB = reg_AB & (dot3(AB, AO) > 0)
            far_AB   = reg_AB & ~close_AB
            d_AB3 = cross3(cross3(AB, AO), AB)
            d_AB3 = torch.where((d_AB3.norm(dim=-1) < EPS).unsqueeze(1), AO, d_AB3)
            d = w3(close_AB, d_AB3, d)
            d = w3(far_AB, AO, d)
            sx[:, 0] = w3(close_AB, Bv3, sx[:, 0])
            sx[:, 1] = w3(close_AB, Av3, sx[:, 1])
            cnt = w1(close_AB, torch.full_like(cnt, 2), cnt)
            sx[:, 0] = w3(far_AB, Av3, sx[:, 0])
            cnt = w1(far_AB, torch.ones_like(cnt), cnt)
            above_tri = reg_tri & (dot3(ABCn, AO) > 0)
            below_tri = reg_tri & ~above_tri
            d = w3(above_tri, ABCn, d)
            d = w3(below_tri, -ABCn, d)
            sx[:, 0] = w3(below_tri, Bv3, sx[:, 0])
            sx[:, 1] = w3(below_tri, Cv3, sx[:, 1])

        m4 = active & (cnt == 4)
        if m4.any():
            Dv4, Cv4, Bv4, Av4 = sx[:, 0], sx[:, 1], sx[:, 2], sx[:, 3]
            AO = -Av4
            n_ABC = cross3(Bv4 - Av4, Cv4 - Av4)
            n_ABC = torch.where((dot3(n_ABC, Dv4 - Av4) > 0).unsqueeze(1), -n_ABC, n_ABC)
            n_ACD = cross3(Cv4 - Av4, Dv4 - Av4)
            n_ACD = torch.where((dot3(n_ACD, Bv4 - Av4) > 0).unsqueeze(1), -n_ACD, n_ACD)
            n_ADB = cross3(Dv4 - Av4, Bv4 - Av4)
            n_ADB = torch.where((dot3(n_ADB, Cv4 - Av4) > 0).unsqueeze(1), -n_ADB, n_ADB)
            abc_up  = m4 & (dot3(n_ABC, AO) > 0)
            acd_up  = m4 & ~abc_up & (dot3(n_ACD, AO) > 0)
            adb_up  = m4 & ~abc_up & ~acd_up & (dot3(n_ADB, AO) > 0)
            inside4 = m4 & ~abc_up & ~acd_up & ~adb_up
            hit    = hit | inside4
            active = active & ~inside4
            sx[:, 0] = w3(abc_up, Cv4, sx[:, 0])
            sx[:, 1] = w3(abc_up, Bv4, sx[:, 1])
            sx[:, 2] = w3(abc_up, Av4, sx[:, 2])
            cnt = w1(abc_up, torch.full_like(cnt, 3), cnt)
            sx[:, 0] = w3(acd_up, Dv4, sx[:, 0])
            sx[:, 1] = w3(acd_up, Cv4, sx[:, 1])
            sx[:, 2] = w3(acd_up, Av4, sx[:, 2])
            cnt = w1(acd_up, torch.full_like(cnt, 3), cnt)
            sx[:, 0] = w3(adb_up, Bv4, sx[:, 0])
            sx[:, 1] = w3(adb_up, Dv4, sx[:, 1])
            sx[:, 2] = w3(adb_up, Av4, sx[:, 2])
            cnt = w1(adb_up, torch.full_like(cnt, 3), cnt)
            d = w3(abc_up, n_ABC, d)
            d = w3(acd_up, n_ACD, d)
            d = w3(adb_up, n_ADB, d)

    return hit | active


def compute_geo_pen_batch(settled_root, settled_quat, child_hull_dev, device):
    B, n_ch = settled_root.shape[0], settled_root.shape[1]
    r = torch.zeros(B, device=device)
    world_hulls = []
    for j in range(n_ch):
        if child_hull_dev[j] is None:
            world_hulls.append(None)
            continue
        pos_j = settled_root[:, j, :].to(device)
        q_j   = settled_quat[:, j, :].to(device)
        R_j   = quat_to_rotmat_batch(q_j)
        V_loc = child_hull_dev[j]
        V_w   = torch.bmm(R_j, V_loc.T.unsqueeze(0).expand(B, -1, -1)).transpose(1, 2) \
                + pos_j.unsqueeze(1)
        world_hulls.append(V_w)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            if world_hulls[i] is None or world_hulls[j] is None:
                continue
            collision = gjk_batch(world_hulls[i], world_hulls[j])
            r += collision.float()
    return r


def compute_geo_pair_flags_single(root_one, quat_one, child_hull_dev, device):
    n_obj = root_one.shape[0]
    world_hulls = []
    for j in range(n_obj):
        if child_hull_dev[j] is None:
            world_hulls.append(None)
            continue
        pos_j = root_one[j:j+1, :].to(device)
        q_j   = quat_one[j:j+1, :].to(device)
        R_j   = quat_to_rotmat_batch(q_j)
        V_loc = child_hull_dev[j]
        V_w   = torch.bmm(R_j, V_loc.T.unsqueeze(0)).transpose(1, 2) + pos_j.unsqueeze(1)
        world_hulls.append(V_w)
    flags = {}
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            if world_hulls[i] is None or world_hulls[j] is None:
                flags[(i, j)] = False
                continue
            flags[(i, j)] = bool(gjk_batch(world_hulls[i], world_hulls[j])[0].item())
    return flags
