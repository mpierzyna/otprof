"""Simulation definition for a single 5-day WRF run over the Canary Islands."""

from wrf_massive.base import BBox, Simulation

mynn25 = {
    # MYNN 2.5
    "physics__sf_sfclay_physics": "5",
    "physics__bl_pbl_physics": "5",
    "physics__bl_mynn_closure": "2.5",
    # without EDMF (so, mixing length 1)
    "physics__bl_mynn_mixlength": "1",
    "physics__bl_mynn_edmf": "0",
    "physics__bl_mynn_edmf_mom": "0",
    "physics__bl_mynn_edmf_tke": "0",
    "physics__sf_urban_physics": "0",  # no urban model
}

# Lat/lon box the CDS forcing download must cover. Fully contains the 300x170 @ 2km Canaries domain
# (centre 28.4N/15.7W) with margin; validated against namelist.tmpl.wps by PullCdsStage.
canaries_area = BBox(north=32.0, west=-20.5, south=24.0, east=-11.0)

# Four 5-day net simulation across seasons
sim_canaries = [
    Simulation(
        begin=f"2020-{month:02d}-01T00:00:00",
        end=f"2020-{month:02d}-06T00:00:00",
        warmup_h=12,
        sim_dir=f"sim_2020-{month:02d}-01",
        settings=mynn25,
        area=canaries_area,
    )
    for month in [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
]


if __name__ == "__main__":
    print(sim_canaries)
