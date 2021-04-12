import torch
import fire
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

def test_kb_sum_multilingual(model_path ='', gpu=-1):
    if gpu >= 0:
        device= torch.device("cuda:"+str(gpu))
    else:
        device= torch.device("cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path) #cpu
    article="""`` BUG # : 149383 ( Content Maintenance ) VSTS : 429231 VSTS : 820756 `` `` Microsoft vertreibt Microsoft SQL Server 2008-Fixes als eine herunterladbare Datei . Da die Fixes kumulativ sind , enthält jede neue Version alle Hotfixes und alle Sicherheitsupdates , die in der vorherigen Version von SQL Server 2008 behoben wurden. `` <s> Problembeschreibung </s> `` Stellen Sie sich folgendes Szenario vor : Sie haben Microsoft SQL Server 2008 oder SQL Server 2008 R2-Datenbankmodul auf einem Computer installiert . Sie erstellen eine Instanz von SQL Server 2008 oder SQL Server 2008 R2 . Sie Slipstream Microsoft SQL Server 2008 Service Pack 1 ( SP1 ) , Microsoft SQL Server 2008 Service Pack 2 ( SP2 ) oder Microsoft SQL Server 2008 R2 Service Pack 1 ( SP1 ) in das ursprüngliche SQL-Installationsmedium . Sie verwenden das slipstreamed-Installationsmedium , um der SQL-Instanz neue Features hinzuzufügen . In diesem Szenario schlägt die Slipstream-Installation möglicherweise fehl. `` <s> Ursache </s> `` Dieses Problem tritt aufgrund eines Fehlers in der Setup Komponente des SQL Server 2008 und des SQL Server 2008 R2-Datenbankmoduls auf . Die Setup-Komponente unterstützt keine Slipstream-Installationen , die neue Features hinzufügen . Wenn eine Slipstream-Installation ausgeführt wird , tritt ein Watson Bucket 482639000-Problem auf , und der Setupvorgang schlägt während der Setup actionName : sqlengineconfigaction_patch_validation Phase fehl. `` <s> Fehlerbehebung </s> `` Service Pack-Informationen für SQL Server 2008 Um dieses Problem zu beheben , besorgen Sie sich das neueste Service Pack für SQL Server 2008 . Weitere Informationen finden Sie im folgenden Artikel der Microsoft Knowledge Base : 968382 So erhalten Sie das neueste Service Pack für SQL Server 2008 Service Pack-Informationen für SQL Server 2008 R2 Um dieses Problem zu beheben , besorgen Sie sich das neueste Service Pack für SQL Server 2008 R2 . Weitere Informationen finden Sie im folgenden Artikel der Microsoft Knowledge Base : 2527041 So erhalten Sie das neueste Service Pack für SQL Server 2008 R2 `` <s> Problemumgehung </s> `` Um dieses Problem zu umgehen , aktualisieren Sie SQL Server 2008 oder SQL Server 2008 R2-Datenbankmodul mithilfe einer anderen Kopie des Service Packs , das sich auf dem Slipstream-Installationsmedium befindet , und fügen Sie dann die neuen Features hinzu. `` <s> Weitere Informationen </s> `` Weitere Informationen zu den häufig gestellten Fragen zu SQL Server 2008 Slipstream finden Sie auf der folgenden MSDN-Website : Häufig gestellte Fragen zu SQL Server 2008 Slipstream Weitere Informationen zum Beheben des SQL Server 2008-Setups vor der Ausführung von Setup finden Sie auf der folgenden MSDN-Website : So beheben Sie das Setup von SQL Server 2008 vor der Ausführung von Setup `` `` Author : mitan ; jannaw Writer : v-allzhu Tech Reviewer : mitan ; jannaw Editor : v-lynan ``
    """
    batch = tokenizer([article], return_tensors="pt", truncation=True, padding="longest").to(device)
    generated_tokens = model.generate(**batch)
    generated_tokens = model.generate(input_ids = batch.input_ids, attention_mask=batch.attention_mask)
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(generated_text)

if __name__ =='__main__':
    fire.Fire(test_kb_sum_multilingual)