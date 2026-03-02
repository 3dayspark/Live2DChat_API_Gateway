import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CharacterAttributes:
    """キャラクターのUI・LLM属性情報を保持するデータクラス (Gateway専用)"""
    def __init__(self):
        self.character_folder_name = ''
        self.character_name = ''
        self.icon_path = None
        self.live2d_json = ''
        self.character_description = ''
        self.rvc_model_dir_id = ''
        self.rvc_index_dir_id = ''

class GetCharacterAttributes:
    """API Gateway内に配置された live2d_related ディレクトリから設定をロードする"""
    def __init__(self):
        self.character_num = 0
        self.character_class_list =[]
        self.load_data()
        
        logger.info('ロードされたキャラクター:')
        for char in self.character_class_list:
            logger.info(f"- {char.character_name} (Folder: {char.character_folder_name})")

    def load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # API_Gatewayプロジェクト内にある live2d_related フォルダを指す
        live2d_base_path = os.path.join(script_dir, "live2d_related")
        
        if not os.path.exists(live2d_base_path):
            raise FileNotFoundError(f"ディレクトリが見つかりません: {live2d_base_path}")

        for char_folder in os.listdir(live2d_base_path):
            full_path = os.path.join(live2d_base_path, char_folder)
            
            if os.path.isdir(full_path):
                self.character_num += 1
                character = CharacterAttributes()
                character.character_folder_name = char_folder

                # キャラクター名
                name_file_path = os.path.join(full_path, 'name.txt')
                if os.path.exists(name_file_path):
                    with open(name_file_path, 'r', encoding='utf-8') as f:
                        character.character_name = f.read().strip()

                # アイコン
                program_icon_paths = glob.glob(os.path.join(full_path, "*.png"))
                if program_icon_paths:
                    character.icon_path = max(program_icon_paths, key=os.path.getmtime)

                # Live2D JSON
                character.live2d_json = f"/models/{character.character_folder_name}/live2D_model/3.model.json"

                # キャラクター設定（プロンプト用）
                desc_file_path = os.path.join(full_path, 'character_description.txt')
                if os.path.exists(desc_file_path):
                    with open(desc_file_path, 'r', encoding='utf-8') as f:
                        character.character_description = f.read()

                # RVC設定用ID（フォルダ名をそのまま使用）
                character.rvc_model_dir_id = os.path.join(character.character_folder_name, "rvc_model")
                character.rvc_index_dir_id = os.path.join(character.character_folder_name, "rvc_model")

                self.character_class_list.append(character)