"""Testing of the cubedsphere GlobalCSBasis class.

1) Conversions to / from cubed sphere, Cartesian, and spherical
coordinates 2) Conversions to / from cubed sphere, Cartesians, and
spherical components 3) Plot cubed sphere grid and vector components
in Cartesian 3D and on Cartopy projection 4) Plot eastward and northward
vector fields on cubed sphere blocks
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from kompe import GlobalCSProjection

p = GlobalCSProjection()


def geocentric_to_plate_carree_vector_components(east, north, latitude):
    """Adjust geocentric components for a Plate Carree test plot."""
    magnitude = np.sqrt(east**2 + north**2)
    east_pc = east / np.cos(np.deg2rad(latitude))
    magnitude_pc = np.sqrt(east_pc**2 + north**2)
    east_pc = east_pc * magnitude / magnitude_pc
    north_pc = north * magnitude / magnitude_pc
    return east_pc, north_pc


def test_projection():
    """Test cubed sphere functionality."""
    # Test conversions to / from cubed sphere, Cartesian, and spherical
    # coordinates.
    N = 5000  # number of test points
    xx, yy, zz = (
        2 * np.random.random(N) - 1,
        2 * np.random.random(N) - 1,
        2 * np.random.random(N) - 1,
    )
    rr = np.sqrt(xx**2 + yy**2 + zz**2)
    inside_unit_sphere = rr <= 1
    rr = rr[inside_unit_sphere]
    xx = xx[inside_unit_sphere]
    yy = yy[inside_unit_sphere]
    zz = zz[inside_unit_sphere]

    lon = np.rad2deg(np.arctan2(yy, xx))
    lat = np.rad2deg(np.arcsin(zz / rr))

    block = p.face_index(lon, lat)
    xi, eta, block = p.geographic_to_cube(lon, lat, block)

    _, theta, phi = p.cube_to_spherical(xi, eta, radius=rr, face=block)
    geo2cube2spherical_works = np.allclose(90 - np.rad2deg(theta) - lat, 0) & np.allclose(
        np.rad2deg(phi) - lon, 0
    )
    print(
        "Conversion from (lon, lat, block) to (xi, eta) and back works: "
        f"{geo2cube2spherical_works}"
    )

    assert geo2cube2spherical_works

    x, y, z = p.cube_to_cartesian(xi, eta, radius=rr, face=block)
    geo2cube2cartesian_works = (
        np.allclose(x - xx, 0) & np.allclose(y - yy, 0) & np.allclose(z - zz, 0)
    )
    print(
        "Conversion from (x, y, z, block ) to (xi, eta) and back works: "
        f"{geo2cube2cartesian_works}"
    )

    assert geo2cube2cartesian_works

    # Test conversions to / from cubed sphere, Cartesian, and spherical
    # components.
    N = xx.size
    Axyz = 2 * np.random.random((3, N)) - 1  # (3, N) random vector components

    Pc = p.cartesian_to_cube_vector_matrix(xi, eta, radius=rr, face=block)
    Pcinv = p.cube_to_cartesian_vector_matrix(xi, eta, radius=rr, face=block)
    enu_to_cube = p.enu_to_cube_vector_matrix(xi, eta, radius=rr, face=block)
    cube_to_enu = p.cube_to_enu_vector_matrix(xi, eta, radius=rr, face=block)

    A = np.einsum("nij, nj -> ni", Pc, Axyz.T).T
    Axyz_ = np.einsum("nij, nj -> ni", Pcinv, A.T).T
    Aenu = np.einsum("nij, nj -> ni", cube_to_enu, A.T).T
    A_ = np.einsum("nij, nj -> ni", enu_to_cube, Aenu.T).T

    cubed2cartesian_works = np.allclose(Axyz - Axyz_, 0)
    print(
        "Converting vector components between cubed sphere and Cartesian give "
        f"consistent results: {cubed2cartesian_works}"
    )
    cubed2enu_works = np.allclose(A_ - A, 0)
    print(
        "Converting vector components between cubed sphere and ENU gives "
        f"consistent results: {cubed2enu_works}"
    )
    norm_consistent = np.allclose(np.linalg.norm(Aenu, axis=0), np.linalg.norm(Axyz, axis=0))
    print(f"Cartesian and ENU vectors have the same norm: {norm_consistent}")

    assert cubed2cartesian_works
    assert cubed2enu_works
    assert norm_consistent

    # Plot cubed sphere grid and vector components in Cartesian 3D and
    # on Cartopy projection.
    print(
        "Plotting cubed sphere grid and vector components in Cartesian 3D and "
        "on Cartopy projection"
    )

    phi0, lat0 = 0, 0
    N = 16  # Number of grid points in each direction per block (should be even)

    fig = plt.figure(figsize=(12, 8))
    axxyz1 = fig.add_subplot(233, projection="3d")
    axxyz2 = fig.add_subplot(236, projection="3d")

    cartopyprojection1 = ccrs.Orthographic(phi0, lat0 + 20)
    cartopyprojection2 = ccrs.Orthographic(phi0 + 180, lat0 + 70)
    axg1 = fig.add_subplot(231, projection=cartopyprojection1)
    axg2 = fig.add_subplot(232, projection=cartopyprojection1)
    axg3 = fig.add_subplot(234, projection=cartopyprojection2)
    axg4 = fig.add_subplot(235, projection=cartopyprojection2)
    for ax in [axg1, axg2, axg3, axg4]:
        ax.coastlines(zorder=3)

    xi, eta = np.meshgrid(
        np.linspace(-np.pi / 4, np.pi / 4, N), np.linspace(-np.pi / 4, np.pi / 4, N), indexing="ij"
    )
    ones = np.ones_like(xi).reshape(-1)
    zeros = np.zeros_like(eta).reshape(-1)
    rs = np.zeros_like(eta).reshape(-1)
    Axis = np.vstack((ones, zeros, rs)).T
    Aetas = np.vstack((zeros, ones, rs)).T

    print("--- some Cartopy / matplotlib warnings:")
    for i in range(6):
        C = "C" + str(i)

        # Plot spherical coordinates using cartopy.
        _, theta, phi = p.cube_to_spherical(xi, eta, face=i)
        lo, la = np.rad2deg(phi), 90 - np.rad2deg(theta)
        lon, lat = np.rad2deg(phi).reshape(-1), 90 - np.rad2deg(theta).reshape(-1)
        cube_to_enu = p.cube_to_enu_vector_matrix(xi, eta, radius=1, face=i)

        # Project in xi-direction.
        Aeast, Anorth, Ar = np.einsum("nij, nj -> ni", cube_to_enu, Axis).T
        assert np.all(np.isclose(Ar, 0))
        # norms = np.sqrt(Aeast**2 + Anorth**2)

        Ae_pc, An_pc = geocentric_to_plate_carree_vector_components(
            Aeast.reshape(-1), Anorth.reshape(-1), lat
        )
        axg1.quiver(lon, lat, Ae_pc, An_pc, transform=ccrs.PlateCarree(), color=C)
        axg2.quiver(lon, lat, Ae_pc, An_pc, transform=ccrs.PlateCarree(), color=C)

        # Project in eta-direction.
        Aeast, Anorth, Ar = np.einsum("nij, nj -> ni", cube_to_enu, Aetas).T
        assert np.all(np.isclose(Ar, 0))

        Ae_pc, An_pc = geocentric_to_plate_carree_vector_components(
            Aeast.reshape(-1), Anorth.reshape(-1), lat
        )
        axg3.quiver(lon % 360, lat, Ae_pc, An_pc, transform=ccrs.PlateCarree(), color=C)
        axg4.quiver(lon % 360, lat, Ae_pc, An_pc, transform=ccrs.PlateCarree(), color=C)

        for ax in [axg1, axg2, axg3, axg4]:
            ax.scatter(lon, lat, color=C, transform=ccrs.PlateCarree(), s=5, zorder=60)

            for k in range(N):
                ax.plot(
                    lo[k, :].reshape(-1),
                    la[k, :].reshape(-1),
                    color=C,
                    linewidth=0.5,
                    linestyle="--",
                    transform=ccrs.Geodetic(),
                )
                ax.plot(
                    lo[:, k].reshape(-1),
                    la[:, k].reshape(-1),
                    color=C,
                    linewidth=0.5,
                    linestyle="--",
                    transform=ccrs.Geodetic(),
                )

        # Plot Cartesian 3D coordinates.
        x, y, z = p.cube_to_cartesian(xi, eta, face=i)
        axxyz1.scatter(x, y, z, c=C, s=5)
        axxyz2.scatter(x, y, z, c=C, s=5)
        Pc = p.cube_to_cartesian_vector_matrix(xi, eta, radius=1, face=i)

        Ax, Ay, Az = np.einsum("nij, nj -> ni", Pc, Axis).T
        axxyz1.quiver(
            x.reshape(-1), y.reshape(-1), z.reshape(-1), Ax, Ay, Az, length=1e-1, color=C
        )

        Ax, Ay, Az = np.einsum("nij, nj -> ni", Pc, Aetas).T
        axxyz2.quiver(
            x.reshape(-1), y.reshape(-1), z.reshape(-1), Ax, Ay, Az, length=1e-1, color=C
        )

    # Make Cartesian plots prettier.
    for ax in [axxyz1, axxyz2]:
        ax.set_axis_off()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        a = 0.95

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = a * np.outer(np.cos(u), np.sin(v))
        y = a * np.outer(np.sin(u), np.sin(v))
        z = a * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color="white")

    # axxyz1.set_title(r'$\xi$ direction')
    # axxyz2.set_title(r'$\eta$ direction')

    # Link two 3D axes so that if one rotates, the other does too.
    def on_move(event):
        if event.inaxes == axxyz2:
            axxyz1.view_init(elev=axxyz2.elev, azim=axxyz2.azim)
        elif event.inaxes == axxyz1:
            axxyz2.view_init(elev=axxyz1.elev, azim=axxyz1.azim)
        else:
            return
        fig.canvas.draw_idle()

    # c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.tight_layout()

    # Plot eastward and northward vector fields on cubed sphere blocks.
    print("Plotting eastward and northward vector fields on cubed sphere blocks")
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(18, 6))

    xihd, etahd = np.meshgrid(
        np.linspace(-np.pi / 4, np.pi / 4, 400), np.linspace(-np.pi / 4, np.pi / 4, 400)
    )

    Aeast = np.vstack((ones, zeros, zeros))
    Anorth = np.vstack((zeros, ones, zeros))

    for block in range(6):
        _, theta_hd, phi_hd = p.cube_to_spherical(xihd, etahd, radius=1, face=block)
        la = 90 - np.rad2deg(theta_hd)
        lo = np.rad2deg(phi_hd)

        _, theta, phi = p.cube_to_spherical(xi, eta, radius=1, face=block)
        enu_to_cube = p.enu_to_cube_vector_matrix(xi, eta, radius=1, face=block)

        Ae = np.einsum("nij, nj -> ni", enu_to_cube, Aeast.T).T
        An = np.einsum("nij, nj -> ni", enu_to_cube, Anorth.T).T
        assert np.allclose(np.hstack((Ae[2], An[2])), 0)

        axes[0, block].scatter(xi, eta, c="grey", zorder=1, s=5)
        axes[1, block].scatter(xi, eta, c="grey", zorder=1, s=5)

        axes[0, block].quiver(xi.reshape(-1), eta.reshape(-1), Ae[0], Ae[1], scale=15)
        axes[1, block].quiver(xi.reshape(-1), eta.reshape(-1), An[0], An[1], scale=15)

        axes[0, block].set_title("block " + str(block) + ", eastward")
        axes[1, block].set_title("block " + str(block) + ", northward")

        for ax in axes.T[block]:
            # cs = ax.contour(
            #    xihd,
            #    etahd,
            #    la,
            #    levels=np.r_[-80:90:10],
            #    colors="lightgrey",
            #    linewidths=1,
            #    zorder=0,
            # )
            # ax.clabel(
            #    cs,
            #    cs.levels,
            #    inline=True,
            #    fmt=lambda x: r"{:.0f}$^\circ$N".format(x),
            #    zorder=0,
            # )
            for lo_ in np.r_[-180:180:30]:
                la_ = np.linspace(-90, 90, 181)
                f = p.face_index(lo_, la_)
                la_ = la_[f == block]

                la_[np.abs(la_) > 80] = np.nan
                xi_, eta_, _ = p.geographic_to_cube(lo_, la_, block)
                ax.plot(xi_, eta_, zorder=0, linewidth=1, color="lightgrey")

    for ax in axes.reshape(-1):
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(-np.pi / 4, np.pi / 4)
        ax.set_ylim(-np.pi / 4, np.pi / 4)

    # Plot grid with ghost cells on a map.
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection=cartopyprojection1)
    ax.coastlines(zorder=3)

    xi, eta = np.meshgrid(
        np.linspace(-np.pi / 4, np.pi / 4, N), np.linspace(-np.pi / 4, np.pi / 4, N)
    )
    dxi = np.diff(xi[0])[0]
    det = np.diff(eta[:, 0])[0]

    N_extra = 5
    extra_xi = np.arange(1, N_extra + 1) * dxi
    extra_eta = np.arange(1, N_extra + 1) * det
    xi_larger = np.hstack((xi[0, 0] - extra_xi[::-1], xi[0, :], xi[0, -1] + extra_xi))
    eta_larger = np.hstack((eta[0, 0] - extra_eta[::-1], eta[:, 0], eta[-1, 0] + extra_eta))
    xi_, eta_ = np.meshgrid(xi_larger, eta_larger)

    for i in range(6):
        # Plot the main grid on each block.
        C = "C" + str(i)

        # Plot spherical coordinates using cartopy.
        _, theta, phi = p.cube_to_spherical(xi, eta, face=i)
        lo, la = np.rad2deg(phi), 90 - np.rad2deg(theta)

        for k in range(N):
            ax.plot(
                lo[k, :].reshape(-1),
                la[k, :].reshape(-1),
                color=C,
                linewidth=0.5,
                linestyle="--",
                transform=ccrs.Geodetic(),
            )
            ax.plot(
                lo[:, k].reshape(-1),
                la[:, k].reshape(-1),
                color=C,
                linewidth=0.5,
                linestyle="--",
                transform=ccrs.Geodetic(),
            )

    _, theta, phi = p.cube_to_spherical(xi_, eta_, face=0)
    lo, la = np.rad2deg(phi), 90 - np.rad2deg(theta)
    for k in range(N + 2 * N_extra):
        ax.plot(
            lo[k, :].reshape(-1),
            la[k, :].reshape(-1),
            color="C0",
            linewidth=1,
            linestyle="-",
            transform=ccrs.Geodetic(),
        )
        ax.plot(
            lo[:, k].reshape(-1),
            la[:, k].reshape(-1),
            color="C0",
            linewidth=1,
            linestyle="-",
            transform=ccrs.Geodetic(),
        )

    # plt.show()
    # plt.close()
