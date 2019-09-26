from data_loaders import SemcorDataLoader


class SemevalDataLoader(SemcorDataLoader):
    def __init__(self, config, xml_path='', gold_path=''):
        super().__init__(config)

        self.xml_path = xml_path
        self.gold_path = gold_path


if __name__ == '__main__':
    pass
