import yaml
from types import SimpleNamespace


# This is used to convert all the keys in the dict into string
convert_keys = lambda d: dict([(str(e[0]), e[1]) for e in d.items()])

class Config:
    """
    Config file parser. It supports the defined
    YAML files only.
    """
    conf = None

    def __init__(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                Config.conf = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise RuntimeError(e)
        self._data = Config.conf.get('data')
        self.logging = Config.conf.get('logging')
        self.type = Config.conf.get('type')
        if self._data is None:
            raise ValueError('Data section in the config file is not found')
        if self.logging is None:
            raise ValueError('Logging section is the config file is not found')
        if self.type.lower() not in ['classification', 'regression', 'generative']:
            raise NotImplementedError('Unknown type for the learning type')

    @property
    def dirs(self):
        return SimpleNamespace(**self._data['dirs'])

    @property
    def labels(self):
        label = self._data['labels']
        return label.get('label_file'), convert_keys(label.get('mappings'))

    @property
    def operations(self):
        return SimpleNamespace(**self._data['ops'])

