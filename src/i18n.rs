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
        "Place ton curseur où tu veux écrire, appuie sur ton raccourci, parle, puis ré-appuie : le texte est transcrit en local et collé automatiquement.\nÉchap annule l'enregistrement en cours.",
        "Place your cursor where you want to type, press your shortcut, speak, then press again: the text is transcribed locally and pasted automatically.\nEsc cancels the current recording.",
        "Coloca el cursor donde quieras escribir, pulsa tu atajo, habla, y vuelve a pulsar: el texto se transcribe localmente y se pega automáticamente.\nEsc cancela la grabación en curso.",
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
    ("status_busy", "Transcription en cours…", "Still transcribing…", "Transcripción en curso…"),
    // --- Tray ---
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
