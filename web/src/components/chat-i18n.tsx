import { cloneElement, isValidElement, type ReactNode } from "react"

export const CHAT_LANGUAGES = ["en", "es", "zh-CN", "hi", "ur", "bn", "fr", "de", "ja", "pt", "it", "uk"] as const
export type ChatLanguage = typeof CHAT_LANGUAGES[number]
export const CHAT_LANGUAGE_STORAGE_KEY = "magnitude.language"
export const CHAT_LANGUAGE_CHANGE_EVENT = "magnitude-language-change"

export const CHAT_LANGUAGE_NAMES: Record<ChatLanguage, string> = {
  en: "English", es: "Español", "zh-CN": "简体中文", hi: "हिन्दी", ur: "اردو",
  bn: "বাংলা", fr: "Français", de: "Deutsch", ja: "日本語", pt: "Português",
  it: "Italiano", uk: "Українська",
}

export const CHAT_TRANSLATION_KEYS = [
  "New chat",
  "Local inference · streaming responses",
  "Model",
  "No models available",
  "Magnitude has no models available for inference yet. Install or load a model, then retry.",
  "Could not load local models: ",
  "Retry",
  "Browser storage is unavailable; history may be lost when you leave this page.",
  "Your conversations will appear here.",
  "Stored in this browser only.",
  "No history",
  "Chat with your local model",
  "Choose a model and ask a question. Text appears as it is generated. Your conversation history stays in this browser.",
  "Load a model in My Models before starting a conversation.",
  "You",
  "Magnitude",
  "Generating…",
  "Could not complete the request: ",
  "The inference response did not provide a stream.",
  "Local Open WebUI URL",
  "Enter a valid local URL using localhost, 127.0.0.1, or [::1]. Remote hosts and embedded credentials are not allowed.",
  "Connect",
  "Only loopback URLs are accepted. The default port 3000 is an example; use your actual Open WebUI address. A blank frame may mean the server blocks embedding with CSP or X-Frame-Options.",
  "Open WebUI",
  "Open in new tab",
  "Back to native Chat",
  "Embedded local interface · external server policy still applies",
  "Connect your local Open WebUI",
  "Enter its loopback URL above. Magnitude will not install or start Open WebUI for you, and this iframe does not bypass the server's embedding policy.",
  "Message your local model…",
  "Enter to send · Shift+Enter for a new line",
  "Send",
  "Stop",
  "Conversation history",
  "Message",
  "Language",
  "Chat language",
  "Delete ",
] as const

const translations: Record<ChatLanguage, readonly string[]> = {
  en: CHAT_TRANSLATION_KEYS,
  es: [
    "Nuevo chat", "Inferencia local · respuestas en streaming", "Modelo", "No hay modelos disponibles",
    "Magnitude todavía no tiene modelos disponibles para inferencia. Instala o carga un modelo y vuelve a intentarlo.",
    "No se pudieron cargar los modelos locales: ", "Reintentar",
    "El almacenamiento del navegador no está disponible; puedes perder el historial al salir de esta página.",
    "Tus conversaciones aparecerán aquí.", "Guardado solo en este navegador.", "Sin historial",
    "Chatea con tu modelo local",
    "Elige un modelo y haz una pregunta. El texto aparece mientras se genera. El historial permanece en este navegador.",
    "Carga un modelo en Mis modelos antes de iniciar una conversación.", "Tú", "Magnitude", "Generando…",
    "No se pudo completar la solicitud: ", "La respuesta de inferencia no proporcionó un flujo.",
    "URL local de Open WebUI",
    "Introduce una URL local válida con localhost, 127.0.0.1 o [::1]. No se permiten hosts remotos ni credenciales incrustadas.",
    "Conectar",
    "Solo se aceptan URL de loopback. El puerto 3000 es un ejemplo; usa la dirección real de Open WebUI. Un marco vacío puede indicar que el servidor bloquea la inserción mediante CSP o X-Frame-Options.",
    "Open WebUI", "Abrir en una pestaña nueva", "Volver al chat nativo",
    "Interfaz local integrada · siguen aplicándose las políticas del servidor externo",
    "Conecta tu Open WebUI local",
    "Introduce arriba su URL de loopback. Magnitude no instalará ni iniciará Open WebUI, y este marco no evita la política de inserción del servidor.",
    "Escribe a tu modelo local…", "Intro para enviar · Mayús+Intro para nueva línea", "Enviar", "Detener",
    "Historial de conversaciones", "Mensaje", "Idioma", "Idioma del chat", "Eliminar ",
  ],
  "zh-CN": [
    "新建聊天", "本地推理 · 流式响应", "模型", "没有可用模型",
    "Magnitude 暂无可用于推理的模型。请安装或加载模型后重试。", "无法加载本地模型：", "重试",
    "浏览器存储不可用；离开此页面后，聊天记录可能丢失。", "你的对话会显示在这里。", "仅存储在此浏览器中。", "没有历史记录",
    "与你的本地模型聊天", "选择一个模型并提出问题。文本会随着生成逐步显示。对话记录保留在此浏览器中。",
    "请先在“我的模型”中加载模型。", "你", "Magnitude", "正在生成…", "无法完成请求：", "推理响应未提供数据流。",
    "本地 Open WebUI URL", "请输入有效的本地 URL：localhost、127.0.0.1 或 [::1]。不允许远程主机或嵌入式凭据。",
    "连接", "仅接受回环地址。默认端口 3000 仅为示例，请使用实际的 Open WebUI 地址。空白框架可能表示服务器通过 CSP 或 X-Frame-Options 禁止嵌入。",
    "Open WebUI", "在新标签页中打开", "返回原生聊天", "嵌入式本地界面 · 仍受外部服务器策略约束",
    "连接本地 Open WebUI", "在上方输入回环 URL。Magnitude 不会为你安装或启动 Open WebUI，此框架也不会绕过服务器的嵌入策略。",
    "向本地模型发送消息…", "按 Enter 发送 · 按 Shift+Enter 换行", "发送", "停止", "对话历史", "消息", "语言", "聊天语言", "删除 ",
  ],
  hi: [
    "नई चैट", "स्थानीय अनुमान · स्ट्रीमिंग प्रतिक्रियाएँ", "मॉडल", "कोई मॉडल उपलब्ध नहीं",
    "Magnitude में अभी अनुमान के लिए कोई मॉडल उपलब्ध नहीं है। मॉडल इंस्टॉल या लोड करके फिर कोशिश करें।", "स्थानीय मॉडल लोड नहीं हो सके: ", "फिर कोशिश करें",
    "ब्राउज़र स्टोरेज उपलब्ध नहीं है; पेज छोड़ने पर इतिहास खो सकता है।", "आपकी बातचीत यहाँ दिखाई देगी।", "केवल इसी ब्राउज़र में सहेजा गया।", "कोई इतिहास नहीं",
    "अपने स्थानीय मॉडल से चैट करें", "मॉडल चुनें और सवाल पूछें। टेक्स्ट बनते समय दिखाई देगा। बातचीत का इतिहास इसी ब्राउज़र में रहता है।",
    "बातचीत शुरू करने से पहले My Models में मॉडल लोड करें।", "आप", "Magnitude", "बन रहा है…", "अनुरोध पूरा नहीं हो सका: ", "इन्फरेंस प्रतिक्रिया में स्ट्रीम नहीं मिली।",
    "स्थानीय Open WebUI URL", "localhost, 127.0.0.1 या [::1] वाला मान्य स्थानीय URL दर्ज करें। दूरस्थ होस्ट और एम्बेड किए गए क्रेडेंशियल स्वीकार नहीं हैं।",
    "कनेक्ट करें", "केवल लूपबैक URL स्वीकार हैं। पोर्ट 3000 उदाहरण है; अपना वास्तविक Open WebUI पता इस्तेमाल करें। खाली फ़्रेम का अर्थ हो सकता है कि सर्वर CSP या X-Frame-Options से एम्बेडिंग रोकता है।",
    "Open WebUI", "नए टैब में खोलें", "नेटिव चैट पर लौटें", "एम्बेड किया गया स्थानीय इंटरफ़ेस · बाहरी सर्वर की नीतियाँ लागू रहेंगी",
    "अपने स्थानीय Open WebUI से कनेक्ट करें", "ऊपर उसका लूपबैक URL डालें। Magnitude आपके लिए Open WebUI इंस्टॉल या शुरू नहीं करेगा और यह फ़्रेम सर्वर की एम्बेडिंग नीति को नहीं बायपास करता।",
    "अपने स्थानीय मॉडल को संदेश दें…", "भेजने के लिए Enter · नई पंक्ति के लिए Shift+Enter", "भेजें", "रोकें", "बातचीत का इतिहास", "संदेश", "भाषा", "चैट की भाषा", "हटाएँ ",
  ],
  ur: [
    "نئی چیٹ", "مقامی اِنفرنس · اسٹریم ہوتی جوابات", "ماڈل", "کوئی ماڈل دستیاب نہیں",
    "Magnitude میں ابھی اِنفرنس کے لیے کوئی ماڈل دستیاب نہیں۔ ماڈل انسٹال یا لوڈ کرکے دوبارہ کوشش کریں۔", "مقامی ماڈل لوڈ نہیں ہو سکے: ", "دوبارہ کوشش",
    "براؤزر اسٹوریج دستیاب نہیں؛ صفحہ چھوڑنے پر تاریخ ضائع ہو سکتی ہے۔", "آپ کی گفتگو یہاں دکھائی دے گی۔", "صرف اسی براؤزر میں محفوظ ہے۔", "کوئی تاریخ نہیں",
    "اپنے مقامی ماڈل سے چیٹ کریں", "ماڈل منتخب کریں اور سوال پوچھیں۔ متن بنتے ہی دکھائی دے گا۔ گفتگو کی تاریخ اسی براؤزر میں رہتی ہے۔",
    "گفتگو شروع کرنے سے پہلے My Models میں ماڈل لوڈ کریں۔", "آپ", "Magnitude", "تیار ہو رہا ہے…", "درخواست مکمل نہیں ہو سکی: ", "اِنفرنس جواب میں اسٹریم فراہم نہیں کی گئی۔",
    "مقامی Open WebUI URL", "localhost، 127.0.0.1 یا [::1] والا درست مقامی URL درج کریں۔ دور دراز ہوسٹس اور شامل کردہ اسناد کی اجازت نہیں۔",
    "کنیکٹ کریں", "صرف لوپ بیک URLs قبول ہیں۔ پورٹ 3000 مثال ہے؛ اپنا اصل Open WebUI پتہ استعمال کریں۔ خالی فریم کا مطلب ہو سکتا ہے کہ سرور CSP یا X-Frame-Options کے ذریعے ایمبیڈنگ روکتا ہے۔",
    "Open WebUI", "نئے ٹیب میں کھولیں", "مقامی چیٹ پر واپس جائیں", "ایمبیڈ کیا گیا مقامی انٹرفیس · بیرونی سرور کی پالیسی لاگو رہے گی",
    "اپنے مقامی Open WebUI سے جڑیں", "اوپر اس کا لوپ بیک URL درج کریں۔ Magnitude آپ کے لیے Open WebUI انسٹال یا شروع نہیں کرے گا اور یہ فریم سرور کی ایمبیڈنگ پالیسی کو نظرانداز نہیں کرتا۔",
    "اپنے مقامی ماڈل کو پیغام دیں…", "بھیجنے کے لیے Enter · نئی سطر کے لیے Shift+Enter", "بھیجیں", "روکیں", "گفتگو کی تاریخ", "پیغام", "زبان", "چیٹ کی زبان", "حذف کریں ",
  ],
  bn: [
    "নতুন চ্যাট", "স্থানীয় ইনফারেন্স · স্ট্রিমিং উত্তর", "মডেল", "কোনো মডেল উপলভ্য নেই",
    "Magnitude-এ ইনফারেন্সের জন্য এখনও কোনো মডেল নেই। মডেল ইনস্টল বা লোড করে আবার চেষ্টা করুন।", "স্থানীয় মডেল লোড করা যায়নি: ", "আবার চেষ্টা করুন",
    "ব্রাউজার স্টোরেজ উপলভ্য নয়; পৃষ্ঠা ছাড়লে ইতিহাস হারিয়ে যেতে পারে।", "আপনার কথোপকথন এখানে দেখা যাবে।", "শুধু এই ব্রাউজারে সংরক্ষিত।", "কোনো ইতিহাস নেই",
    "আপনার স্থানীয় মডেলের সঙ্গে চ্যাট করুন", "একটি মডেল বেছে প্রশ্ন করুন। তৈরি হওয়ার সঙ্গে সঙ্গে লেখা দেখা যাবে। কথোপকথনের ইতিহাস এই ব্রাউজারেই থাকে।",
    "কথোপকথন শুরু করার আগে My Models-এ একটি মডেল লোড করুন।", "আপনি", "Magnitude", "তৈরি হচ্ছে…", "অনুরোধ সম্পন্ন করা যায়নি: ", "ইনফারেন্স প্রতিক্রিয়ায় কোনো স্ট্রিম ছিল না।",
    "স্থানীয় Open WebUI URL", "localhost, 127.0.0.1 বা [::1] ব্যবহার করে বৈধ স্থানীয় URL দিন। দূরবর্তী হোস্ট ও এমবেড করা পরিচয়পত্র অনুমোদিত নয়।",
    "সংযোগ করুন", "শুধু লুপব্যাক URL গ্রহণ করা হয়। পোর্ট 3000 উদাহরণ; আপনার আসল Open WebUI ঠিকানা ব্যবহার করুন। ফাঁকা ফ্রেমের অর্থ সার্ভার CSP বা X-Frame-Options দিয়ে এমবেডিং আটকাচ্ছে।",
    "Open WebUI", "নতুন ট্যাবে খুলুন", "নেটিভ চ্যাটে ফিরুন", "এমবেড করা স্থানীয় ইন্টারফেস · বাহ্যিক সার্ভারের নীতি প্রযোজ্য",
    "আপনার স্থানীয় Open WebUI-তে সংযোগ করুন", "উপরে লুপব্যাক URL দিন। Magnitude আপনার জন্য Open WebUI ইনস্টল বা চালু করবে না এবং এই ফ্রেম সার্ভারের এমবেডিং নীতি এড়ায় না।",
    "আপনার স্থানীয় মডেলকে বার্তা দিন…", "পাঠাতে Enter · নতুন লাইন Shift+Enter", "পাঠান", "থামান", "কথোপকথনের ইতিহাস", "বার্তা", "ভাষা", "চ্যাটের ভাষা", "মুছুন ",
  ],
  fr: [
    "Nouvelle discussion", "Inférence locale · réponses en continu", "Modèle", "Aucun modèle disponible",
    "Magnitude ne dispose pas encore de modèle pour l’inférence. Installez ou chargez un modèle, puis réessayez.", "Impossible de charger les modèles locaux : ", "Réessayer",
    "Le stockage du navigateur est indisponible ; l’historique peut être perdu en quittant cette page.", "Vos conversations apparaîtront ici.", "Stocké uniquement dans ce navigateur.", "Aucun historique",
    "Discutez avec votre modèle local", "Choisissez un modèle et posez une question. Le texte apparaît au fur et à mesure. L’historique reste dans ce navigateur.",
    "Chargez un modèle dans Mes modèles avant de commencer.", "Vous", "Magnitude", "Génération…", "Impossible de terminer la requête : ", "La réponse d’inférence ne fournit aucun flux.",
    "URL locale d’Open WebUI", "Saisissez une URL locale valide avec localhost, 127.0.0.1 ou [::1]. Les hôtes distants et identifiants intégrés sont interdits.",
    "Connecter", "Seules les URL de bouclage sont acceptées. Le port 3000 est un exemple ; utilisez l’adresse réelle d’Open WebUI. Un cadre vide peut indiquer un blocage par CSP ou X-Frame-Options.",
    "Open WebUI", "Ouvrir dans un nouvel onglet", "Revenir au chat natif", "Interface locale intégrée · la politique du serveur externe reste applicable",
    "Connecter votre Open WebUI local", "Saisissez son URL de bouclage ci-dessus. Magnitude n’installera ni ne démarrera Open WebUI, et ce cadre ne contourne pas la politique d’intégration du serveur.",
    "Écrivez à votre modèle local…", "Entrée pour envoyer · Maj+Entrée pour un saut de ligne", "Envoyer", "Arrêter", "Historique des conversations", "Message", "Langue", "Langue du chat", "Supprimer ",
  ],
  de: [
    "Neuer Chat", "Lokale Inferenz · Streaming-Antworten", "Modell", "Keine Modelle verfügbar",
    "Magnitude hat noch keine Modelle für die Inferenz. Installiere oder lade ein Modell und versuche es erneut.", "Lokale Modelle konnten nicht geladen werden: ", "Erneut versuchen",
    "Der Browserspeicher ist nicht verfügbar; der Verlauf kann beim Verlassen dieser Seite verloren gehen.", "Deine Unterhaltungen erscheinen hier.", "Nur in diesem Browser gespeichert.", "Kein Verlauf",
    "Chatte mit deinem lokalen Modell", "Wähle ein Modell und stelle eine Frage. Der Text erscheint während der Generierung. Der Verlauf bleibt in diesem Browser.",
    "Lade zuerst unter Meine Modelle ein Modell.", "Du", "Magnitude", "Generierung…", "Anfrage konnte nicht abgeschlossen werden: ", "Die Inferenzantwort hat keinen Stream geliefert.",
    "Lokale Open-WebUI-URL", "Gib eine gültige lokale URL mit localhost, 127.0.0.1 oder [::1] ein. Entfernte Hosts und eingebettete Zugangsdaten sind nicht erlaubt.",
    "Verbinden", "Nur Loopback-URLs sind zulässig. Port 3000 ist ein Beispiel; verwende die tatsächliche Open-WebUI-Adresse. Ein leerer Rahmen kann durch CSP oder X-Frame-Options verursacht werden.",
    "Open WebUI", "In neuem Tab öffnen", "Zur nativen Chat-Oberfläche", "Eingebettete lokale Oberfläche · Richtlinien des externen Servers gelten weiterhin",
    "Lokale Open WebUI verbinden", "Gib oben die Loopback-URL ein. Magnitude installiert oder startet Open WebUI nicht für dich; dieser Rahmen umgeht die Einbettungsrichtlinie des Servers nicht.",
    "Nachricht an dein lokales Modell…", "Enter zum Senden · Umschalt+Enter für neue Zeile", "Senden", "Stoppen", "Unterhaltungsverlauf", "Nachricht", "Sprache", "Chat-Sprache", "Löschen ",
  ],
  ja: [
    "新しいチャット", "ローカル推論 · ストリーミング応答", "モデル", "利用可能なモデルがありません",
    "Magnitude には推論に利用できるモデルがまだありません。モデルをインストールまたは読み込んで再試行してください。", "ローカルモデルを読み込めませんでした: ", "再試行",
    "ブラウザーのストレージを利用できません。このページを離れると履歴が失われる可能性があります。", "会話がここに表示されます。", "このブラウザーにのみ保存されます。", "履歴はありません",
    "ローカルモデルとチャット", "モデルを選んで質問してください。生成中のテキストが表示されます。会話履歴はこのブラウザーに保存されます。",
    "会話を始める前に「My Models」でモデルを読み込んでください。", "あなた", "Magnitude", "生成中…", "リクエストを完了できませんでした: ", "推論応答にストリームがありません。",
    "ローカル Open WebUI URL", "localhost、127.0.0.1、または [::1] を使った有効なローカル URL を入力してください。リモートホストや埋め込み認証情報は許可されません。",
    "接続", "ループバック URL のみ利用できます。ポート 3000 は例です。実際の Open WebUI アドレスを使用してください。空白のフレームは CSP または X-Frame-Options による埋め込み拒否を示す場合があります。",
    "Open WebUI", "新しいタブで開く", "ネイティブチャットに戻る", "埋め込みローカル UI · 外部サーバーのポリシーが適用されます",
    "ローカル Open WebUI に接続", "上にループバック URL を入力してください。Magnitude は Open WebUI をインストールまたは起動せず、このフレームもサーバーの埋め込みポリシーを回避しません。",
    "ローカルモデルにメッセージ…", "Enter で送信 · Shift+Enter で改行", "送信", "停止", "会話履歴", "メッセージ", "言語", "チャットの言語", "削除 ",
  ],
  pt: [
    "Novo chat", "Inferência local · respostas em streaming", "Modelo", "Nenhum modelo disponível",
    "O Magnitude ainda não tem modelos disponíveis para inferência. Instale ou carregue um modelo e tente novamente.", "Não foi possível carregar os modelos locais: ", "Tentar novamente",
    "O armazenamento do navegador não está disponível; o histórico pode ser perdido ao sair desta página.", "As suas conversas aparecerão aqui.", "Guardado apenas neste navegador.", "Sem histórico",
    "Converse com o seu modelo local", "Escolha um modelo e faça uma pergunta. O texto aparece à medida que é gerado. O histórico fica neste navegador.",
    "Carregue um modelo em Meus modelos antes de iniciar uma conversa.", "Você", "Magnitude", "A gerar…", "Não foi possível concluir o pedido: ", "A resposta de inferência não forneceu um fluxo.",
    "URL local do Open WebUI", "Introduza um URL local válido com localhost, 127.0.0.1 ou [::1]. Hosts remotos e credenciais incorporadas não são permitidos.",
    "Ligar", "Só são aceites URLs de loopback. A porta 3000 é um exemplo; use o endereço real do Open WebUI. Um quadro vazio pode indicar bloqueio por CSP ou X-Frame-Options.",
    "Open WebUI", "Abrir num novo separador", "Voltar ao chat nativo", "Interface local incorporada · a política do servidor externo continua a aplicar-se",
    "Ligue o seu Open WebUI local", "Introduza acima o URL de loopback. O Magnitude não instalará nem iniciará o Open WebUI e este quadro não contorna a política de incorporação do servidor.",
    "Envie uma mensagem ao seu modelo local…", "Enter para enviar · Shift+Enter para nova linha", "Enviar", "Parar", "Histórico de conversas", "Mensagem", "Idioma", "Idioma do chat", "Eliminar ",
  ],
  it: [
    "Nuova chat", "Inferenza locale · risposte in streaming", "Modello", "Nessun modello disponibile",
    "Magnitude non dispone ancora di modelli per l'inferenza. Installa o carica un modello e riprova.", "Impossibile caricare i modelli locali: ", "Riprova",
    "Lo spazio di archiviazione del browser non è disponibile; la cronologia potrebbe andare persa uscendo da questa pagina.", "Le tue conversazioni appariranno qui.", "Salvato solo in questo browser.", "Nessuna cronologia",
    "Chatta con il tuo modello locale", "Scegli un modello e fai una domanda. Il testo appare durante la generazione. La cronologia resta in questo browser.",
    "Carica un modello in I miei modelli prima di iniziare una conversazione.", "Tu", "Magnitude", "Generazione…", "Impossibile completare la richiesta: ", "La risposta di inferenza non ha fornito uno stream.",
    "URL locale di Open WebUI", "Inserisci un URL locale valido con localhost, 127.0.0.1 o [::1]. Host remoti e credenziali incorporate non sono consentiti.",
    "Connetti", "Sono accettati solo URL loopback. La porta 3000 è un esempio; usa l'indirizzo effettivo di Open WebUI. Un riquadro vuoto può indicare un blocco tramite CSP o X-Frame-Options.",
    "Open WebUI", "Apri in una nuova scheda", "Torna alla chat nativa", "Interfaccia locale incorporata · si applicano ancora le regole del server esterno",
    "Connetti il tuo Open WebUI locale", "Inserisci sopra il suo URL loopback. Magnitude non installerà né avvierà Open WebUI e questo riquadro non aggira la politica di incorporamento del server.",
    "Scrivi al tuo modello locale…", "Invio per inviare · Maiusc+Invio per andare a capo", "Invia", "Interrompi", "Cronologia conversazioni", "Messaggio", "Lingua", "Lingua della chat", "Elimina ",
  ],
  uk: [
    "Новий чат", "Локальний висновок · потокові відповіді", "Модель", "Немає доступних моделей",
    "У Magnitude поки немає моделей для інференсу. Установіть або завантажте модель і повторіть спробу.", "Не вдалося завантажити локальні моделі: ", "Повторити",
    "Сховище браузера недоступне; історія може втратитися після виходу зі сторінки.", "Ваші розмови з’являться тут.", "Збережено лише в цьому браузері.", "Історії немає",
    "Спілкуйтеся з локальною моделлю", "Виберіть модель і поставте запитання. Текст з’являється під час генерації. Історія зберігається в цьому браузері.",
    "Перш ніж почати, завантажте модель у розділі «Мої моделі».", "Ви", "Magnitude", "Генерація…", "Не вдалося виконати запит: ", "Відповідь інференсу не містила потоку.",
    "Локальна URL-адреса Open WebUI", "Укажіть коректну локальну URL-адресу з localhost, 127.0.0.1 або [::1]. Віддалені хости та вбудовані облікові дані заборонені.",
    "Підключити", "Дозволені лише loopback URL. Порт 3000 наведено як приклад; використовуйте фактичну адресу Open WebUI. Порожній фрейм може означати блокування вбудовування через CSP або X-Frame-Options.",
    "Open WebUI", "Відкрити в новій вкладці", "Повернутися до вбудованого чату", "Вбудований локальний інтерфейс · політика зовнішнього сервера залишається чинною",
    "Підключіть локальний Open WebUI", "Введіть вище loopback URL. Magnitude не встановлюватиме й не запускатиме Open WebUI, а цей фрейм не обходить політику вбудовування сервера.",
    "Напишіть локальній моделі…", "Enter — надіслати · Shift+Enter — новий рядок", "Надіслати", "Зупинити", "Історія розмов", "Повідомлення", "Мова", "Мова чату", "Видалити ",
  ],
}

export const CHAT_TRANSLATION_COUNTS = Object.fromEntries(
  CHAT_LANGUAGES.map(language => [language, translations[language].length]),
) as Record<ChatLanguage, number>

const dictionaries = Object.fromEntries(
  CHAT_LANGUAGES.map(language => [language, Object.fromEntries(CHAT_TRANSLATION_KEYS.map((key, index) => [key, translations[language][index] ?? key]))]),
) as Record<ChatLanguage, Record<string, string>>

export function isChatLanguage(value: unknown): value is ChatLanguage {
  return typeof value === "string" && (CHAT_LANGUAGES as readonly string[]).includes(value)
}

export function readChatLanguage(): ChatLanguage {
  try {
    const value = typeof localStorage === "undefined" ? null : localStorage.getItem(CHAT_LANGUAGE_STORAGE_KEY)
    return isChatLanguage(value) ? value : "en"
  } catch {
    return "en"
  }
}

export function writeChatLanguage(language: ChatLanguage): void {
  try { localStorage.setItem(CHAT_LANGUAGE_STORAGE_KEY, language) } catch { /* Storage can be disabled. */ }
  if (typeof window !== "undefined") window.dispatchEvent(new Event(CHAT_LANGUAGE_CHANGE_EVENT))
}

export function translateChatText(value: string, language: ChatLanguage): string {
  const dictionary = dictionaries[language]
  if (dictionary[value] !== undefined) return dictionary[value]
  for (const prefix of ["Could not load local models: ", "Could not complete the request: ", "Delete "]) {
    if (value.startsWith(prefix)) return (dictionary[prefix] ?? prefix) + value.slice(prefix.length)
  }
  return value
}

/** Translate only UI copy and attributes; keep user/assistant message content and model IDs untouched. */
export function localizeChatTree(node: ReactNode, language: ChatLanguage, insideMessage = false): ReactNode {
  if (typeof node === "string") return insideMessage ? node : translateChatText(node, language)
  if (Array.isArray(node)) return node.map(child => localizeChatTree(child, language, insideMessage))
  if (!isValidElement(node)) return node
  const props = node.props as Record<string, unknown>
  const isMessage = insideMessage || node.type === "article"
  const nextProps: Record<string, unknown> = { ...props }
  for (const key of ["aria-label", "placeholder", "title"]) {
    if (typeof nextProps[key] === "string") nextProps[key] = translateChatText(nextProps[key] as string, language)
  }
  if ("children" in props) {
    nextProps.children = localizeChatTree(props.children as ReactNode, language, isMessage)
  }
  return cloneElement(node, nextProps)
}
