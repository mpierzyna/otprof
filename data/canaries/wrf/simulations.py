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

# Lat/lon box the CDS forcing download must cover. Fully contains the Canaries domain with margin.
canaries_area = BBox(north=32.0, west=-20.5, south=24.0, east=-11.0)

# One 5-day net simulation per month. Use 2017 because testing in OTProf training.
sim_canaries = [
    Simulation(
        begin=f"2017-{month:02d}-01T00:00:00",
        end=f"2017-{month:02d}-06T00:00:00",
        warmup_h=12,
        sim_dir=f"sim_2017-{month:02d}-01",
        settings=mynn25,
        area=canaries_area,
    )
    for month in range(1, 13)  # one sim per month
]


if __name__ == "__main__":
    print(sim_canaries)
