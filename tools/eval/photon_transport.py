"""Independent monochromatic planar transport verifier, not an 800T material.

MCML-style optical-depth hopping, implicit absorption, HG scattering and Fresnel
boundaries. Lengths are um, mu_a/mu_s are 1/um. Packets carry optical energy;
neither film sensitivity nor developed dye/base density enters this module.
No roulette: small/trapped/event-limited weights are explicitly unresolved.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layer:
    name: str
    thickness_um: float
    mu_a: float
    mu_s: float
    g: float = 0.0
    n: float = 1.5

    def __post_init__(self):
        values = (self.thickness_um, self.mu_a, self.mu_s, self.g, self.n)
        if not np.isfinite(values).all():
            raise ValueError('Layer parameters must be finite')
        if self.thickness_um <= 0 or min(self.mu_a, self.mu_s) < 0 or self.n <= 0:
            raise ValueError('Positive thickness/index and nonnegative coefficients required')
        if abs(self.g) >= 1:
            raise ValueError('HG anisotropy must be strictly between -1 and 1')


def fresnel(cosine, n_from, n_to):
    """Unpolarized power reflectance and transmitted absolute direction cosine."""
    ci, ni, nt = np.broadcast_arrays(np.asarray(cosine, float), n_from, n_to)
    if (not all(np.isfinite(v).all() for v in (ci, ni, nt))
            or np.any((ci < 0) | (ci > 1)) or np.any(ni <= 0) or np.any(nt <= 0)):
        raise ValueError('Invalid cosine or refractive index')
    sin2 = (ni / nt)**2 * np.maximum(0., 1 - ci**2)
    ct = np.sqrt(np.maximum(0., 1 - sin2))
    rs = np.divide(ni*ci-nt*ct, ni*ci+nt*ct,
                   out=np.ones_like(ci), where=ni*ci+nt*ct > 0)
    rp = np.divide(nt*ci-ni*ct, nt*ci+ni*ct,
                   out=np.ones_like(ci), where=nt*ci+ni*ct > 0)
    reflectance = np.where(sin2 >= 1, 1., (rs**2 + rp**2)/2)
    reflectance = np.where(ni == nt, 0., reflectance)
    return reflectance, ct


def spin(directions, g, rng):
    """Draw HG deflections relative to each incoming unit direction."""
    g = np.broadcast_to(np.asarray(g, float), (len(directions),))
    u = rng.random(len(directions))
    c = 2*u - 1
    nonzero = np.abs(g) > 1e-12
    gn = g[nonzero]
    c[nonzero] = (1 + gn**2 - ((1-gn**2)/(1-gn+2*gn*u[nonzero]))**2)/(2*gn)
    c = np.clip(c, -1, 1)
    s = np.sqrt(1-c*c)
    phi = 2*np.pi*rng.random(len(directions))
    # Stable orthonormal basis, including directions parallel to z.
    helper = np.zeros_like(directions)
    helper[:, 2] = 1
    helper[np.abs(directions[:, 2]) > .9] = (1, 0, 0)
    u_axis = np.cross(helper, directions)
    u_axis /= np.linalg.norm(u_axis, axis=1)[:, None]
    v_axis = np.cross(directions, u_axis)
    result = (directions*c[:, None] + u_axis*(s*np.cos(phi))[:, None]
              + v_axis*(s*np.sin(phi))[:, None])
    return result / np.linalg.norm(result, axis=1)[:, None]


def simulate(layers, photons=100_000, seed=0, *, n_top=1., n_bottom=1.,
             incident_cosine=1., radial_edges=None, max_events=4000,
             weight_cutoff=1e-9):
    """Normally/obliquely incident monochromatic pencil beam into planar layers.

    Entrance reflection is split deterministically; internal interfaces use
    analog reflection/refraction. Remaining optical depth survives boundaries.
    'returned' absorption means after reflection at the final rear interface,
    not a separate physical source or extra energy. Histogram overflow is kept.
    """
    layers = tuple(layers)
    if not layers or photons < 1 or int(photons) != photons or max_events < 1:
        raise ValueError('Need layers, integer photons > 0 and max_events > 0')
    if not 0 < incident_cosine <= 1 or not 0 <= weight_cutoff < 1:
        raise ValueError('Invalid incident cosine or weight cutoff')
    if not np.isfinite([n_top, n_bottom]).all() or min(n_top, n_bottom) <= 0:
        raise ValueError('External indices must be finite and positive')
    photons = int(photons)
    edges = np.asarray(radial_edges if radial_edges is not None else np.arange(0, 5010, 10), float)
    if len(edges) < 2 or edges[0] != 0 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('Finite radial edges must start at zero and increase')
    rng = np.random.default_rng(seed)
    nl = len(layers)
    z_edges = np.r_[0., np.cumsum([l.thickness_um for l in layers])]
    ma = np.array([l.mu_a for l in layers]); ms = np.array([l.mu_s for l in layers])
    mut = ma + ms
    indices = np.array([l.n for l in layers]); gs = np.array([l.g for l in layers])
    rsp, ct = fresnel(incident_cosine, n_top, indices[0])
    w = np.full(photons, 1-float(rsp))
    position = np.zeros((photons, 3))
    direction = np.tile([np.sqrt(max(0., 1-float(ct)**2)), 0., float(ct)], (photons, 1))
    layer = np.zeros(photons, int)
    tau = rng.exponential(size=photons)
    alive = w > 0
    returned = np.zeros(photons, bool)
    absorbed = np.zeros(nl); absorbed_returned = np.zeros(nl)
    hist = np.zeros((nl, len(edges)-1)); hist_returned = np.zeros_like(hist)
    overflow = np.zeros(nl); overflow_returned = np.zeros(nl)
    top_escape = bottom_escape = cutoff_weight = 0.
    events = 0
    for events in range(1, int(max_events)+1):
        ids = np.flatnonzero(alive)
        if not len(ids):
            break
        li = layer[ids]; uz = direction[ids, 2]
        boundary_z = np.where(uz > 0, z_edges[li+1], z_edges[li])
        db = np.divide(boundary_z-position[ids, 2], uz,
                       out=np.full(len(ids), np.inf), where=uz != 0)
        db = np.maximum(db, 0.)
        dc = np.divide(tau[ids], mut[li], out=np.full(len(ids), np.inf), where=mut[li] > 0)
        distance = np.minimum(db, dc)
        finite = np.isfinite(distance)
        if not finite.all():
            # Nonpropagating packets remain in the unresolved ledger below.
            alive[ids[~finite]] = False
            ids, li, db, dc, distance = [a[finite] for a in (ids, li, db, dc, distance)]
        position[ids] += direction[ids]*distance[:, None]
        tau[ids] = np.maximum(0., tau[ids]-mut[li]*distance)
        collision = dc < db
        hit = ids[collision]; hl = li[collision]
        if len(hit):
            # Divide first: w*ma/mut can round above w in a pure absorber.
            drop = np.minimum(w[hit], w[hit]*(ma[hl]/mut[hl]))
            w[hit] -= drop
            np.add.at(absorbed, hl, drop)
            np.add.at(absorbed_returned, hl, drop*returned[hit])
            radius = np.linalg.norm(position[hit, :2], axis=1)
            bins = np.searchsorted(edges, radius, side='right')-1
            inside = bins < len(edges)-1
            np.add.at(hist, (hl[inside], bins[inside]), drop[inside])
            np.add.at(hist_returned, (hl[inside], bins[inside]), drop[inside]*returned[hit[inside]])
            np.add.at(overflow, hl[~inside], drop[~inside])
            np.add.at(overflow_returned, hl[~inside], drop[~inside]*returned[hit[~inside]])
            small = w[hit] <= weight_cutoff
            cutoff_weight += w[hit[small]].sum()
            w[hit[small]] = 0.; alive[hit[small]] = False
            scatter = hit[~small]
            if len(scatter):
                direction[scatter] = spin(direction[scatter], gs[layer[scatter]], rng)
                tau[scatter] = rng.exponential(size=len(scatter))
        cross = ids[~collision]; cl = li[~collision]
        if len(cross):
            sign = np.where(direction[cross, 2] > 0, 1, -1)
            target = cl+sign
            ni = indices[cl]
            nt = indices[np.clip(target, 0, nl-1)].copy()
            nt[target < 0] = n_top; nt[target >= nl] = n_bottom
            r, ct = fresnel(np.abs(direction[cross, 2]), ni, nt)
            reflect = rng.random(len(cross)) < r
            ri = cross[reflect]
            direction[ri, 2] *= -1
            returned[ri] |= (target[reflect] == nl)
            ti = cross[~reflect]; dest = target[~reflect]
            direction[ti, :2] *= (ni[~reflect]/nt[~reflect])[:, None]
            direction[ti, 2] = sign[~reflect]*ct[~reflect]
            upper, lower = dest < 0, dest >= nl
            top_escape += w[ti[upper]].sum(); bottom_escape += w[ti[lower]].sum()
            escaped = ti[upper | lower]
            w[escaped] = 0.; alive[escaped] = False
            layer[ti[~(upper | lower)]] = dest[~(upper | lower)]
    unresolved = w.sum()/photons
    ledger = dict(specular_reflection=float(rsp), top_escape=top_escape/photons,
                  bottom_escape=bottom_escape/photons, absorbed=absorbed.sum()/photons,
                  cutoff_residual=cutoff_weight/photons, event_residual=unresolved)
    ledger['closure_error'] = sum(ledger.values())-1.
    return dict(ledger=ledger, absorbed=absorbed/photons,
                returned_absorbed=absorbed_returned/photons,
                radial_absorbed=hist/photons, radial_returned=hist_returned/photons,
                radial_overflow=overflow/photons, returned_overflow=overflow_returned/photons,
                radial_edges=edges, photons=photons, events=events, seed=seed)


def angle_return(optical_depth_passive, optical_depth_active, *, count=100_000,
                 base_um=130., forward=1.5, n_base=1.5, angle_dependent=True):
    """One-return analytic energy ledger for a prescribed internal angular source.

    Radius is at the base's front return plane, not absorption depth in the
    emulsion. Passive absorption precedes active absorption on the return path.
    Normal-depth comparator deliberately omits oblique attenuation and capture.
    No scattering coefficient is inferred from this prescribed angular source.
    """
    params = (optical_depth_passive, optical_depth_active, base_um, forward, n_base)
    if (not np.isfinite(params).all() or min(optical_depth_passive, optical_depth_active) < 0
            or min(base_um, n_base) <= 0 or forward <= -1 or count < 2 or int(count) != count):
        raise ValueError('Invalid optical depths, angular exponent or quadrature count')
    mu = (np.arange(count)+.5)/count
    weights = (forward+1)*mu**forward/count
    weights /= weights.sum()
    reflectance, _ = fresnel(mu, n_base, 1.)
    path = 1/mu if angle_dependent else np.ones_like(mu)
    tp = np.exp(-optical_depth_passive*path)
    ta = np.exp(-optical_depth_active*path)
    active = weights*reflectance*tp*(1-ta)
    ledger = dict(bottom_escape=float(weights@(1-reflectance)),
                  passive_absorption=float(np.sum(weights*reflectance*(1-tp))),
                  active_absorption=float(active.sum()),
                  top_escape=float(np.sum(weights*reflectance*tp*ta)))
    ledger['closure_error'] = sum(ledger.values())-1
    return dict(radius=2*base_um*np.sqrt(1-mu*mu)/mu, energy=active, ledger=ledger)
