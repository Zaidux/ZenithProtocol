# src/local_agent/LocalExecutionAgent.py

import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class LocalExecutionAgent:
    """
    The Local Execution Agent (LEA) is a specialized module that interacts with
    the user's local operating system to perform tasks. It operates with a strong
    emphasis on user consent, verification, and safety.
    """
    def __init__(self, ckg, arlc, em):
        """
        Initializes the agent with references to core Zenith components for
        knowledge grounding, decision-making, and explainability.
        """
        self.ckg = ckg
        self.arlc = arlc
        self.em = em
        self.temp_storage_path = os.path.join(os.path.expanduser('~'), '.zenith_temp')
        os.makedirs(self.temp_storage_path, exist_ok=True)
        print("Local Execution Agent initialized. Temporary storage created.")

    def _generate_confirmation_prompt(self, plan: Dict) -> Dict:
        """
        Generates a clear, human-readable confirmation prompt before execution.
        """
        action = plan.get("action", "an unspecified action")
        folder_name = plan.get("folder_name", "a new folder")
        source_dir = plan.get("source_directory", "your default download folder")
        
        prompt_text = (
            f"I understand you want me to perform the following action:\n"
            f"1. Create a new folder named '{folder_name}'.\n"
            f"2. Find new photos in '{source_dir}' of you and your dog.\n"
            f"3. Move these photos into the new folder.\n"
            "Is this correct? Please confirm with 'yes' or 'no'."
        )
        return {"prompt": prompt_text, "action_plan": plan}

    def execute_action_plan(self, plan: Dict) -> Dict:
        """
        Executes a multi-step, verified action plan on the local device.
        This process includes a self-validation and self-correction loop.
        """
        # Step 1: User Confirmation (This would be handled by the UI/Chatbot)
        confirmation_data = self._generate_confirmation_prompt(plan)
        print(f"[LEA] Awaiting user confirmation with prompt: '{confirmation_data['prompt']}'")
        # For this code, we'll assume confirmation is granted.
        permission_granted = True
        if not permission_granted:
            return {"status": "aborted", "message": "Execution was aborted by the user."}

        # Step 2: Create the target folder
        target_dir = os.path.join(os.path.expanduser('~'), 'Desktop', plan['folder_name'])
        os.makedirs(target_dir, exist_ok=True)
        print(f"[LEA] Target folder '{target_dir}' created.")

        # Step 3: Identify and filter files with conceptual understanding
        # The agent uses its conceptual knowledge to perform a nuanced search.
        files_to_move = self._find_and_filter_files(
            source_dir=os.path.join(os.path.expanduser('~'), 'Downloads'),
            file_types=plan['file_types'],
            conceptual_filters=plan['conceptual_filters']
        )
        if not files_to_move:
            self.arlc.report_failure("file_not_found", "No files matched the conceptual filters.")
            return {"status": "failed", "message": "No files were found matching the criteria."}

        # Step 4: Self-Validation (critical for safety)
        # The agent uses its vision model to validate the files before moving.
        validation_errors = self._self_validate_files(files_to_move, plan['conceptual_filters'])
        if validation_errors:
            self.arlc.report_failure("validation_error", "Some files did not match the conceptual criteria.")
            # Trigger self-correction to handle the errors
            # ARLC can then decide whether to ask the user or remove the files from the plan.
            # For this example, we'll proceed but log the errors.
            print("[LEA] Validation errors detected, proceeding with confirmed files.")
        
        # Step 5: Execute the move and log the action
        final_moved_files = []
        try:
            for file_path in files_to_move:
                if file_path not in validation_errors: # Only move validated files
                    shutil.move(file_path, target_dir)
                    final_moved_files.append(file_path)
            
            # Log the successful action to the CKG for an auditable record
            self.em.log_agent_action(
                action_type="move_files",
                details={
                    "source": os.path.join(os.path.expanduser('~'), 'Downloads'),
                    "target": target_dir,
                    "files_moved": final_moved_files
                },
                result="success"
            )
            
            return {"status": "success", "files_moved": final_moved_files, "errors": validation_errors}
        except Exception as e:
            # Self-correction: Save the data in a temporary, secure place
            self._save_for_recovery(final_moved_files)
            self.arlc.report_failure("execution_error", f"An error occurred during file move: {e}")
            return {"status": "failed", "message": f"Execution error: {e}"}

    def _find_and_filter_files(self, source_dir: str, file_types: List[str], conceptual_filters: Dict) -> List[str]:
        """
        Finds files based on type and applies conceptual filters (e.g., 'newly added').
        """
        found_files = []
        now = datetime.now()
        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            if os.path.isfile(file_path) and any(file_path.endswith(f) for f in file_types):
                # Apply conceptual filter for 'newly added'
                if 'newly_added' in conceptual_filters and conceptual_filters['newly_added']:
                    last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (now - last_modified) < timedelta(minutes=30):
                        found_files.append(file_path)
                else:
                    found_files.append(file_path)
        return found_files

    def _self_validate_files(self, file_paths: List[str], conceptual_filters: Dict) -> List[str]:
        """
        Simulates the AI's internal validation of files before a final move.
        This would use the AI's vision model to confirm the content.
        """
        print("[LEA] Validating files with internal vision model...")
        validation_errors = []
        for file_path in file_paths:
            # Mock vision model output. In a real system, this would be a call to the visual encoder.
            if 'dog' in conceptual_filters and 'user' in conceptual_filters:
                # Mock a failure for demonstration
                if 'bad_photo' in file_path:
                    validation_errors.append(file_path)
        return validation_errors
        
    def _save_for_recovery(self, files: List[str]):
        """
        In case of a critical error, saves the files to a temporary location for recovery.
        """
        for file_path in files:
            try:
                shutil.copy(file_path, self.temp_storage_path)
                print(f"[LEA] Recovered '{file_path}' to temporary storage.")
            except Exception as e:
                print(f"[LEA] Could not recover '{file_path}': {e}")
      
