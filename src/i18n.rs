//! Minimal internationalization (FR / EN / ES), port of `freewhisper/i18n.py`.
//!
//! String table + `tr()` function. The active language is global state
//! (`set_lang`); the immediate-mode UI (egui) reflects the change on the next
//! frame, without rebuilding.

use std::sync::atomic::{AtomicU8, Ordering};

static LANG: AtomicU8 = AtomicU8::new(0); // 0=fr, 1=en, 2=es

pub fn set_lang(lang: &str) {
    let v = match lang {
        "en" => 1,
        "es" => 2,
        _ => 0,
    };
    LANG.store(v, Ordering::Relaxed);
}

pub fn get_lang() -> &'static str {
    match LANG.load(Ordering::Relaxed) {
        1 => "en",
        2 => "es",
        _ => "fr",
    }
}

/// Translates `key` into the active language (fallback: French, then the key).
pub fn tr(key: &str) -> &'static str {
    let idx = LANG.load(Ordering::Relaxed) as usize;
    for &(k, fr, en, es) in STRINGS {
        if k == key {
            return [fr, en, es][idx.min(2)];
        }
    }
    "??"
}

// (key, fr, en, es)
static STRINGS: &[(&str, &str, &str, &str)] = &[
    // --- general ---
    ("window_title", "Dictata — Réglages", "Dictata — Settings", "Dictata — Ajustes"),
    ("brand_sub", "100 % local", "100% local", "100% local"),
    ("btn_close", "Fermer", "Close", "Cerrar"),
    ("btn_save", "Enregistrer", "Save", "Guardar"),
    ("saved_ok", "Enregistré ✔", "Saved ✔", "Guardado ✔"),
    // --- navigation ---
    ("nav_home", "Accueil", "Home", "Inicio"),
    ("nav_modes", "Modes", "Modes", "Modos"),
    ("nav_vocab", "Vocabulaire", "Vocabulary", "Vocabulario"),
    ("nav_config", "Configuration", "Configuration", "Configuración"),
    ("nav_sound", "Son", "Sound", "Sonido"),
    ("nav_models", "Modèles", "Models", "Modelos"),
    ("nav_llm", "LLM local", "Local LLM", "LLM local"),
    ("nav_history", "Historique", "History", "Historial"),
    // --- Home ---
    ("home_hw", "Matériel", "Hardware", "Hardware"),
    ("home_reco", "Modèle recommandé :", "Recommended model:", "Modelo recomendado:"),
    ("home_state", "État", "Status", "Estado"),
    ("home_active_model", "Modèle actif :", "Active model:", "Modelo activo:"),
    ("home_active_mode", "Mode actif :", "Active mode:", "Modo activo:"),
    ("home_models_folder", "Dossier des modèles :", "Models folder:", "Carpeta de modelos:"),
    ("home_open_folder", "Ouvrir le dossier des modèles", "Open models folder", "Abrir la carpeta de modelos"),
    ("home_howto_title", "Comment ça marche", "How it works", "Cómo funciona"),
    (
        "home_howto_text",
        "Place ton curseur où tu veux écrire, appuie sur ton raccourci, parle, puis ré-appuie : le texte est transcrit en local et collé automatiquement.\nMaintiens Échap une demi-seconde pour annuler l'enregistrement en cours.",
        "Place your cursor where you want to type, press your shortcut, speak, then press again: the text is transcribed locally and pasted automatically.\nHold Esc for half a second to cancel the current recording.",
        "Coloca el cursor donde quieras escribir, pulsa tu atajo, habla, y vuelve a pulsar: el texto se transcribe localmente y se pega automáticamente.\nMantén Esc medio segundo para cancelar la grabación en curso.",
    ),
    // --- Modes ---
    ("modes_active", "Mode actif", "Active mode", "Modo activo"),
    ("modes_active_hint", "Utilisé pour la dictée.", "Used for dictation.", "Usado para el dictado."),
    ("modes_add", "+ Ajouter", "+ Add", "+ Añadir"),
    ("modes_del", "− Supprimer", "− Remove", "− Eliminar"),
    ("modes_label", "Libellé", "Label", "Etiqueta"),
    ("modes_type", "Type", "Type", "Tipo"),
    ("modes_task", "Tâche", "Task", "Tarea"),
    ("modes_prompt_ph", "Prompt LLM (modes de type llm)", "LLM prompt (llm-type modes)", "Prompt LLM (modos de tipo llm)"),
    ("modes_none_sel", "Aucun mode sélectionné.", "No mode selected.", "Ningún modo seleccionado."),
    ("modes_key_ph", "clé", "key", "clave"),
    // --- Vocabulary ---
    ("vocab_title", "Vocabulaire", "Vocabulary", "Vocabulario"),
    (
        "vocab_hint",
        "Un terme par ligne — aide Whisper à bien orthographier les noms propres, le jargon, etc.",
        "One term per line — helps Whisper spell proper nouns, jargon, etc.",
        "Un término por línea — ayuda a Whisper a escribir bien nombres propios, jerga, etc.",
    ),
    ("repl_title", "Remplacements", "Replacements", "Reemplazos"),
    (
        "repl_hint",
        "Une règle par ligne, format :  à remplacer = par ceci  (insensible à la casse).",
        "One rule per line, format:  to replace = with this  (case-insensitive).",
        "Una regla por línea, formato:  a reemplazar = por esto  (sin distinción de mayúsculas).",
    ),
    // --- Configuration ---
    ("cfg_card", "Raccourci & activation", "Shortcut & activation", "Atajo y activación"),
    ("cfg_hotkey", "Raccourci global", "Global shortcut", "Atajo global"),
    ("cfg_hotkey_hint", "Démarre / arrête la dictée.", "Starts / stops dictation.", "Inicia / detiene el dictado."),
    ("cfg_activation", "Activation", "Activation", "Activación"),
    ("cfg_activation_toggle", "Toggle (appuyer / ré-appuyer)", "Toggle (press / press again)", "Alternar (pulsar / volver a pulsar)"),
    ("cfg_activation_ptt", "Push-to-talk (maintenir)", "Push-to-talk (hold)", "Pulsar para hablar (mantener)"),
    ("cfg_cancel", "Annuler l'enregistrement", "Cancel recording", "Cancelar la grabación"),
    ("cfg_cancel_hint", "Abandonne la prise en cours.", "Discards the active recording.", "Descarta la grabación en curso."),
    ("cfg_autopaste", "Collage automatique", "Auto-paste", "Pegado automático"),
    ("cfg_autopaste_hint", "Colle le texte (Ctrl+V) dans l'application active.", "Pastes the text (Ctrl+V) into the active app.", "Pega el texto (Ctrl+V) en la aplicación activa."),
    ("cfg_streaming", "Mode continu (streaming)", "Continuous mode (streaming)", "Modo continuo (streaming)"),
    (
        "cfg_streaming_hint",
        "Insère le texte au fil de la parole, à chaque pause. Mode Raw uniquement.",
        "Inserts text as you speak, at every pause. Raw mode only.",
        "Inserta el texto mientras hablas, en cada pausa. Solo modo Raw.",
    ),
    ("cfg_vad", "Détection de voix (VAD)", "Voice detection (VAD)", "Detección de voz (VAD)"),
    (
        "cfg_vad_hint",
        "Ignore les silences avant transcription (moins de calcul, moins d'hallucinations). Hors mode continu. Télécharge un petit modèle au 1er usage.",
        "Skips silence before transcription (less compute, fewer hallucinations). Not in continuous mode. Downloads a small model on first use.",
        "Omite los silencios antes de transcribir (menos cómputo, menos alucinaciones). No en modo continuo. Descarga un modelo pequeño la 1ª vez.",
    ),
    ("cfg_lowvoice", "Boost voix basse", "Soft-voice boost", "Refuerzo de voz baja"),
    (
        "cfg_lowvoice_hint",
        "Amplifie les prises à faible volume avant transcription (parler doucement).",
        "Amplifies low-volume takes before transcription (speaking softly).",
        "Amplifica las tomas de bajo volumen antes de transcribir (hablar bajito).",
    ),
    ("cfg_automode_card", "Auto-mode par app", "Auto-mode by app", "Auto-modo por app"),
    ("cfg_automode", "Choisir le mode selon l'app active", "Pick the mode from the active app", "Elegir el modo según la app activa"),
    (
        "cfg_automode_hint",
        "Au démarrage de la dictée, sélectionne le mode associé à l'application active (Windows).",
        "When dictation starts, selects the mode mapped to the active application (Windows).",
        "Al iniciar el dictado, selecciona el modo asociado a la aplicación activa (Windows).",
    ),
    (
        "cfg_automode_map_hint",
        "Une règle par ligne :  motif = mode. Le motif correspond au nom de l'exécutable (outlook.exe = email) ou à un bout du titre de la fenêtre (gmail = email), utile pour distinguer deux onglets du même navigateur. La première règle qui correspond gagne : placez les plus précises en haut. Le nom du mode doit exister dans Modes.",
        "One rule per line:  pattern = mode. The pattern matches the executable name (outlook.exe = email) or any part of the window title (gmail = email), which is how you tell two tabs of the same browser apart. The first matching rule wins, so put the specific ones on top. The mode name must exist in Modes.",
        "Una regla por línea:  patrón = modo. El patrón coincide con el nombre del ejecutable (outlook.exe = email) o con parte del título de la ventana (gmail = email), útil para distinguir dos pestañas del mismo navegador. Gana la primera regla que coincide: coloca arriba las más específicas. El modo debe existir en Modos.",
    ),
    (
        "cfg_automode_last",
        "Dernière fenêtre détectée (exécutable | titre) — copiez-en un morceau pour écrire une règle :",
        "Last detected window (executable | title) — copy a piece of it to write a rule:",
        "Última ventana detectada (ejecutable | título) — copia un fragmento para escribir una regla:",
    ),
    (
        "cfg_automode_llm_warn",
        "Les modes de type « llm » n'auront d'effet qu'une fois le LLM local activé (onglet LLM local).",
        "'llm'-type modes only take effect once the local LLM is enabled (Local LLM tab).",
        "Los modos de tipo «llm» solo tendrán efecto al activar el LLM local (pestaña LLM local).",
    ),
    ("cfg_ui_lang", "Langue de l'interface", "Interface language", "Idioma de la interfaz"),
    ("cfg_ui_lang_hint", "Change la langue de cette fenêtre.", "Changes the language of this window.", "Cambia el idioma de esta ventana."),
    ("cfg_dock_card", "Dock flottant", "Floating dock", "Dock flotante"),
    ("cfg_dock_size", "Taille du dock", "Dock size", "Tamaño del dock"),
    ("cfg_dock_opacity", "Opacité", "Opacity", "Opacidad"),
    ("cfg_dock_position_btn", "Positionner le dock", "Reposition dock", "Reposicionar el dock"),
    (
        "cfg_dock_position_hint",
        "Affiche le dock quelques secondes : glisse-le où tu veux à l'écran.",
        "Shows the dock for a few seconds: drag it anywhere on screen.",
        "Muestra el dock unos segundos: arrástralo a donde quieras en la pantalla.",
    ),
    ("cfg_dock_reset_btn", "Réinitialiser", "Reset", "Restablecer"),
    ("cfg_dock_reset_done", "Position du dock réinitialisée", "Dock position reset", "Posición del dock restablecida"),
    ("cfg_dock_saved", "Position du dock enregistrée ✔", "Dock position saved ✔", "Posición del dock guardada ✔"),
    // --- ShortcutEdit ---
    ("shortcut_press_key", "Appuyez sur une combinaison…", "Press a combination…", "Pulsa una combinación…"),
    ("shortcut_click_edit", "clic pour modifier", "click to edit", "clic para editar"),
    // --- Sound ---
    ("sound_card", "Entrée audio", "Audio input", "Entrada de audio"),
    ("sound_mic", "Microphone", "Microphone", "Micrófono"),
    ("sound_default_mic", "Micro par défaut", "Default microphone", "Micrófono predeterminado"),
    ("sound_beeps", "Sons de début / fin", "Start / end sounds", "Sonidos de inicio / fin"),
    ("sound_beeps_hint", "Bip court au démarrage et à la fin.", "Short beep at the start and end.", "Pitido corto al inicio y al final."),
    ("source_label", "Source d'enregistrement", "Recording source", "Fuente de grabación"),
    (
        "source_hint",
        "Audio système : capture ce qui sort des haut-parleurs (réunions Teams, Discord…).",
        "System audio: captures what plays on your speakers (Teams meetings, Discord…).",
        "Audio del sistema: captura lo que suena en los altavoces (reuniones de Teams, Discord…).",
    ),
    ("source_mic", "Microphone", "Microphone", "Micrófono"),
    ("source_system", "Audio système", "System audio", "Audio del sistema"),
    ("source_mix", "Micro + audio système (réunion)", "Mic + system audio (meeting)", "Micro + audio del sistema (reunión)"),
    // --- Models ---
    ("models_params", "Paramètres du modèle", "Model parameters", "Parámetros del modelo"),
    ("models_default_lang", "Langue par défaut", "Default language", "Idioma por defecto"),
    ("models_accel", "Accélération", "Acceleration", "Aceleración"),
    ("models_accel_hint", "auto = GPU si dispo, sinon CPU.", "auto = GPU if available, else CPU.", "auto = GPU si está disponible, si no CPU."),
    ("models_beam", "Beam size", "Beam size", "Beam size"),
    ("models_beam_hint", "Plus haut = un peu plus précis mais plus lent.", "Higher = slightly more accurate but slower.", "Más alto = un poco más preciso pero más lento."),
    ("models_installed", "Modèles installés", "Installed models", "Modelos instalados"),
    ("models_none", "Aucun modèle téléchargé pour l'instant.", "No model downloaded yet.", "Ningún modelo descargado todavía."),
    ("models_lib", "Bibliothèque de modèles", "Model library", "Biblioteca de modelos"),
    (
        "models_reco_hint",
        "large-v3-turbo = équivalent local de l'« Ultra ». Tailles approximatives.",
        "large-v3-turbo = local equivalent of “Ultra”. Approximate sizes.",
        "large-v3-turbo = equivalente local del «Ultra». Tamaños aproximados.",
    ),
    ("models_use", "Utiliser", "Use", "Usar"),
    ("models_download", "Télécharger", "Download", "Descargar"),
    ("models_active", "✔ Actif", "✔ Active", "✔ Activo"),
    ("models_downloading", "Téléchargement :", "Downloading:", "Descargando:"),
    ("models_done", "Terminé", "Done", "Hecho"),
    ("models_installed_ok", "Modèle installé", "Model installed", "Modelo instalado"),
    ("models_dl_error", "Échec du téléchargement :", "Download failed:", "Error de descarga:"),
    ("models_delete", "Supprimer", "Delete", "Eliminar"),
    ("models_deleted", "Modèle supprimé", "Model deleted", "Modelo eliminado"),
    ("models_del_error", "Échec de la suppression :", "Delete failed:", "Error al eliminar:"),
    // --- HuggingFace search ---
    ("hf_card", "Recherche HuggingFace", "HuggingFace search", "Búsqueda HuggingFace"),
    (
        "hf_hint",
        "URL directe, dépôt (auteur/nom) ou mot-clé. Fichiers ggml .bin uniquement.",
        "Direct URL, repo (owner/name) or keyword. ggml .bin files only.",
        "URL directa, repositorio (autor/nombre) o palabra clave. Solo archivos ggml .bin.",
    ),
    ("hf_search", "Rechercher", "Search", "Buscar"),
    ("hf_searching", "Recherche…", "Searching…", "Buscando…"),
    ("hf_no_results", "Aucun résultat", "No results", "Sin resultados"),
    ("hf_no_bin", "Aucun fichier .bin dans ce dépôt", "No .bin file in this repo", "Ningún archivo .bin en este repositorio"),
    ("hf_browse", "Voir les fichiers", "Browse files", "Ver archivos"),
    // --- LLM ---
    ("llm_card", "Serveur local (OpenAI-compatible)", "Local server (OpenAI-compatible)", "Servidor local (compatible con OpenAI)"),
    ("llm_enable", "Activer le reformatage par LLM", "Enable LLM reformatting", "Activar el reformateo por LLM"),
    ("llm_enable_hint", "Les modes 'llm' (Email, Message…) reformatent le texte.", "'llm' modes (Email, Message…) reformat the text.", "Los modos 'llm' (Email, Mensaje…) reformatean el texto."),
    ("llm_url", "URL locale", "Local URL", "URL local"),
    ("llm_url_hint", "LM Studio, Ollama… (jamais de cloud).", "LM Studio, Ollama… (never cloud).", "LM Studio, Ollama… (nunca en la nube)."),
    ("llm_model", "Modèle", "Model", "Modelo"),
    ("llm_temp", "Température", "Temperature", "Temperatura"),
    ("llm_test", "Tester la connexion", "Test connection", "Probar la conexión"),
    ("llm_ok", "✔ disponible", "✔ available", "✔ disponible"),
    ("llm_ko", "✗ injoignable", "✗ unreachable", "✗ inaccesible"),
    // --- History ---
    ("hist_refresh", "Rafraîchir", "Refresh", "Actualizar"),
    ("hist_clear", "Vider", "Clear", "Vaciar"),
    (
        "llm_remote_warn",
        "Cette adresse n'est pas sur cette machine : le texte de chaque dictée sera envoyé à ce serveur.",
        "This address is not on this machine: the text of every dictation will be sent to that server.",
        "Esta dirección no está en esta máquina: el texto de cada dictado se enviará a ese servidor.",
    ),
    ("hist_enable", "Conserver l'historique", "Keep history", "Conservar el historial"),
    (
        "hist_enable_hint",
        "Chaque dictée est enregistrée en clair dans history.jsonl, à côté de l'application. Désactive si tu préfères ne rien conserver.",
        "Every dictation is stored in clear text in history.jsonl, next to the application. Turn off if you would rather keep nothing.",
        "Cada dictado se guarda en texto plano en history.jsonl, junto a la aplicación. Desactiva si prefieres no conservar nada.",
    ),
    ("hist_limit", "Entrées conservées", "Entries kept", "Entradas conservadas"),
    (
        "hist_limit_hint",
        "Au-delà de ce nombre, les entrées les plus anciennes sont supprimées.",
        "Past this number, the oldest entries are deleted.",
        "Por encima de este número, se eliminan las entradas más antiguas.",
    ),
    ("hist_hint", "Clic sur une ligne pour copier le texte.", "Click a line to copy the text.", "Clic en una línea para copiar el texto."),
    ("hist_none", "Aucune transcription pour l'instant.", "No transcription yet.", "Ninguna transcripción todavía."),
    ("hist_copy_hover", "Cliquer pour copier", "Click to copy", "Clic para copiar"),
    ("hist_copied", "Copié dans le presse-papiers", "Copied to clipboard", "Copiado al portapapeles"),
    ("hist_from_file", "Transcrire un fichier…", "Transcribe a file…", "Transcribir un archivo…"),
    ("filetx_running", "Transcription du fichier…", "Transcribing file…", "Transcribiendo archivo…"),
    ("filetx_done", "Fichier transcrit ✔ (copié)", "File transcribed ✔ (copied)", "Archivo transcrito ✔ (copiado)"),
    ("filetx_error", "Échec de la transcription du fichier", "File transcription failed", "Error al transcribir el archivo"),
    // --- Dock / statuses ---
    ("dock_drag", "Glisse-moi", "Drag me", "Arrástrame"),
    ("status_pasted", "Collé", "Pasted", "Pegado"),
    ("status_reformulated", "Reformulé", "Reformatted", "Reformateado"),
    ("status_raw_fallback", "Collé (brut)", "Pasted (raw)", "Pegado (bruto)"),
    ("status_empty", "(vide)", "(empty)", "(vacío)"),
    ("status_error", "Erreur", "Error", "Error"),
    ("status_paste_error", "Erreur collage", "Paste failed", "Error al pegar"),
    ("status_mic_ko", "Micro KO", "Mic error", "Error de micro"),
    ("status_mic_lost", "Micro perdu", "Mic lost", "Micro perdido"),
    (
        "status_model_recovered",
        "Modèle illisible — retour au modèle par défaut",
        "Unreadable model — reverted to the default one",
        "Modelo ilegible — se volvió al modelo por defecto",
    ),
    ("status_busy", "Transcription en cours…", "Still transcribing…", "Transcripción en curso…"),
    ("status_cancelled", "Annulé", "Cancelled", "Cancelado"),
    // --- Tray ---
    ("tray_mode", "Mode", "Mode", "Modo"),
    ("tray_settings", "Réglages…", "Settings…", "Ajustes…"),
    ("tray_quit", "Quitter", "Quit", "Salir"),
    // --- transcription languages ---
    ("lang_auto", "Auto (détection)", "Auto (detect)", "Auto (detección)"),
    ("lang_fr", "Français", "French", "Francés"),
    ("lang_en", "Anglais", "English", "Inglés"),
    ("lang_es", "Espagnol", "Spanish", "Español"),
    ("lang_de", "Allemand", "German", "Alemán"),
    ("lang_it", "Italien", "Italian", "Italiano"),
    ("lang_pt", "Portugais", "Portuguese", "Portugués"),
    ("lang_nl", "Néerlandais", "Dutch", "Neerlandés"),
    ("lang_ru", "Russe", "Russian", "Ruso"),
    ("lang_zh", "Chinois", "Chinese", "Chino"),
    ("lang_ja", "Japonais", "Japanese", "Japonés"),
    ("lang_ko", "Coréen", "Korean", "Coreano"),
    ("lang_ar", "Arabe", "Arabic", "Árabe"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_and_langs() {
        set_lang("fr");
        assert_eq!(tr("btn_save"), "Enregistrer");
        set_lang("en");
        assert_eq!(tr("btn_save"), "Save");
        set_lang("es");
        assert_eq!(tr("btn_save"), "Guardar");
        assert_eq!(tr("nope"), "??");
        set_lang("zz");
        assert_eq!(tr("btn_save"), "Enregistrer");
        set_lang("fr");
    }
}
