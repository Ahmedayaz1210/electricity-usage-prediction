import panel as pn
import os
from matplotlib import image as mpimg
import matplotlib.pyplot as plt

# Initialize Panel extension
pn.extension()

# Path to the datasets directory (use raw string or forward slashes)
base_path = "./../datasets"

# List of states and years
states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", 
          "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", 
          "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", 
          "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", 
          "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
          "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

years = [2022, 2023, 2024, 2025]

# Dropdown widgets for state and year
state_selector = pn.widgets.Select(name='Select State', options=states)
year_selector = pn.widgets.Select(name='Select Year', options=years)

# Panel to display the image
image_pane = pn.pane.Matplotlib()

# Panel to display error messages
error_pane = pn.pane.Str("")

# Function to update the image based on user selection
def update_image(event):
    state = state_selector.value
    year = year_selector.value
    
    # Construct the image file path
    image_path = os.path.join(base_path, f"state_plots_{year}", f"{state}_{year}.png")
    
    if os.path.exists(image_path):
        try:
            # Read and display the image
            img = mpimg.imread(image_path)
            if img is not None:
                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axis('off')  # Turn off axis labels
                image_pane.object = fig
                error_pane.object = ""  # Clear any error messages
            else:
                error_pane.object = f"Image could not be loaded: {image_path}"
        except Exception as e:
            # Handle any exceptions and display them
            error_pane.object = f"Error loading image: {str(e)}"
            print(f"Error loading image: {str(e)}")
    else:
        # Show a message if the image doesn't exist
        image_pane.object = None
        error_pane.object = f"No image available for {state} in {year}"

# Link the dropdowns to the update function
state_selector.param.watch(update_image, 'value')
year_selector.param.watch(update_image, 'value')

# Initial update to show an image
update_image(None)

info_panel = pn.pane.Markdown("""
# Electricity Usage Dashboard
This dashboard allows you to explore predicted and actual electricity usage data by state. Use the selector to choose a state & year.
""", width=300)

# Layout the widgets, image, and error message
dashboard = pn.Row(
    pn.Column(info_panel, state_selector, year_selector),  # Left side
    pn.Column(image_pane, error_pane)  # Right side
)


# Serve the dashboard
dashboard.show()

