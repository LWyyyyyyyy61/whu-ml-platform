from django.core.files.storage import FileSystemStorage
import os
class MyCustomStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        # Check if the file with the same name already exists
        if self.exists(name):
            # Split the base name and the extension
            base_name, ext = os.path.splitext(name)
            
            # Initialize counter for the numbering
            counter = 1
            
            # Generate a new name until it's unique
            while self.exists(f"{base_name}_{counter}{ext}"):
                counter += 1
                
            # Return the new unique name
            return f"{base_name}_{counter}{ext}"
        
        # If the file does not exist, just return the original name
        return name