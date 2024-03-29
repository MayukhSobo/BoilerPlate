from core.config import Config
from core import *
import click


@click.group()
@click.pass_context
def main(ctx):
    ctx.obj['CONF'] = Config


@main.command()
@click.option(
    '-c',
    '--config-file',
    default='config.yaml',
    type=click.Path(exists=True, resolve_path=True),
    help='Path to the config file that loads the configuration for the project'
)
@click.pass_context
def load(ctx, config_file):
    """
    It loads the configuration files and process the data
    gathered from the data directories mentioned in the
    config file accordingly.

    :param ctx: Click contextual object that holds the config class
    :param config_file: Config file path
    :return: None
    """
    conf_obj = ctx.obj['CONF']
    conf, datafiles = _load(config_file, conf_obj=conf_obj)
    print()
    click.echo('*' * 50)
    click.echo(f'Found {len(datafiles[0])} files for training')
    click.echo(f'Found {len(datafiles[1])} files for testing')
    click.echo('*' * 50)
    print()
    _save_and_store(conf, datafiles, parser)
    # // TODO: Display the file stats if mentioned



def _load(config_file, **kwargs):
    """
    Same as ```load``` but used to call
    from the API for jupyter notebook

    :param config_file: Name of the config file
    :return: List of train and test image paths -> tuple
    """
    conf_obj = kwargs.get('conf_obj', None)
    if conf_obj is None:
        conf_obj = Config
    conf = conf_obj(config_file=config_file)
    # Get all path of the images for both training and testing
    datafiles = gather_data(conf.dirs)
    return conf, datafiles

def _save_and_store(conf, datafiles, parser):
    # split and store all the data into separate directories
    split_and_store(dconf=conf.dirs,
                    oconf=conf.operations,
                    lconf=conf.labels,
                    datafiles=datafiles,
                    parser=parser)


if __name__ == '__main__':
    main(obj={})
