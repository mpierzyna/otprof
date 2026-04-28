from typing import Tuple
import click
import pathlib
import shutil


@click.command()
@click.argument("sim_dirs", nargs=-1, type=click.Path(exists=True, file_okay=False, dir_okay=True))
def gather(sim_dirs: Tuple[str,]):
    for d in sim_dirs:
        d = pathlib.Path(d)
        wps_dir = d / "2_wps"
        if wps_dir.exists() and wps_dir.is_symlink():
            wps_dir_ram = wps_dir.resolve()
            print(f"Moving {wps_dir_ram} to {d}")
            wps_dir.unlink()
            shutil.move(str(wps_dir_ram), str(wps_dir))
        else:
            print(f"No ramdisk found in {d}, skipping")
            continue


if __name__ == "__main__":
    gather()
