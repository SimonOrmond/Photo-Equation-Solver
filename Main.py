from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivy.uix.screenmanager import ScreenManager, NoTransition
from kivymd.uix.hero import MDBoxLayout
from kivymd.uix.screen import MDScreen
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivy.uix.image import Image
from kivymd.uix.imagelist import MDSmartTile
from kivymd.uix.expansionpanel import MDExpansionPanel, MDExpansionPanelOneLine
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserListView
from plyer import filechooser
from kivy.core.window import Window
import matplotlib.pyplot as plt
import cmath
from kivy.uix.popup import Popup
import numpy as np
import easyocr
import re   
import glob
import time
import os
import cv2
import csv

Window.size = (375, 625)

class Login(MDScreen): # creates a Login class the inherits MDScreen
    dialog = None   
    def __init__(self, **kwargs): # initialises class attributes
        super(Login, self).__init__(**kwargs) 
        self.attempt_counter = 0 
    def logger(self): # called when "Login" button pressed
        global username # allows me to use these variables in other functions(only ones that run afters this one)
        global password # ...
        username = self.ids.username.text # sets username variable to value in corresponding text field
        password = self.ids.password.text # ^^... password 
        successful_login = False
        
        with open('logindetails.csv', mode='r') as old_file: # opens csv file to be read from
            csv_reader = csv.reader(old_file)
            for line in csv_reader: # iterates through the lines
                if line==[username, password]: # user is allowed access to app if condition is met
                    MDApp.get_running_app().root.current = 'main' # changes the screen to the 'main' screen
                    self.ids.username.text = "" # empties the contents of the username and password textfields
                    self.ids.password.text = ""
                    m = self.manager.get_screen('main')
                    m.account_screen(username) # passes username to the account screen function
                    successful_login = True
                    self.attempt_counter = 0
            if successful_login == False: 
                self.attempt_counter += 1
                tries = 5-self.attempt_counter
                self.ids.attempt.text = f"       you have {tries} attempts remaining" # if password is wrong, dialogue error message shows  
                error_title = "Error"
                error_message = "Username or Password was incorrect"
                self.show_alert_dialog(error_title, error_message)
            if self.attempt_counter >= 5: # if login button pressed more than 5 times
                  # displays image
                quit() # closes program
        pass
    def show_alert_dialog(self,message_title, message_text): # show dialogue function
        if not self.dialog: 
            self.dialog = MDDialog( # creates dialogue instance
                title=message_title, # formatting of the error box
                text= message_text,
                buttons=[
                    MDFlatButton( # button with box to go back to login screen
                        text="go back",
                        theme_text_color="Custom",
                        on_release=lambda _: self.dialog.dismiss() # closes dialogue box when button pressed
                    ),                    
                ],
            )
        self.dialog.open()

class Main(MDScreen): #Creates the Main class inheriting MDScreen
    def __init__(self, **kwargs): # initiates class
        from kivy.uix.image import Image # locally imports the Image Module 
        super().__init__(**kwargs)     
        self.mycamera = self.ids.camera # calls the camera widget from the .kv file
        self.myimage = Image() # sets the myimage variable to have image properties
        
    def capture(self): # function for taking the picture
        path_to_images = "images"
        timenow = time.strftime("%Y%m%d_%H%M%S") # gets the current time
        self.mycamera.export_to_png("images\image_{}.png".format(timenow)) # exports taken photo to png format with current time
        self.myimage.source = "images\image_{}.png".format(timenow) # saves it to myimage variable
        list_of_images = os.listdir(path_to_images) # uses os module to create a list of all the images in the directory
        if len(list_of_images) > 4: # ensures that directory can only have max 4 images to save computer resource
            files = glob.glob(os.path.join(path_to_images, '*'))
            files.sort(key=os.path.getmtime)          
            os.remove(files[0]) 
        self.img_preprocess(self.myimage.source, upload=False) # passes the image into the img_preprocess function as a paremeter
            
    def upload_photo(self):
        file_chooser = FileChooserListView(filters=["*.png", "*.jpg", "*.jpeg"]) # opens filechooser widget
        file_chooser.bind(on_submit=self.selected)
        self.popup = Popup(title="Select a Photo", content=file_chooser, size_hint=(0.9, 0.9))
        self.popup.open() # opens file chooser widget
    
    def selected(self, instance, selection, *args):
        if selection:
            selected_file = selection[0]
            #self.ids.img_display.source = selected_file
            # Handle the selected file as needed
            self.img_preprocess(selected_file, upload=True)
            self.popup.dismiss() # closes popup after the file is selected
        
        
    def switch_camera(self): # function for rotating the camera, called upon a button press in the .kv file
        camera = self.ids.camera # connects the camera in .kv file to .py script
        if camera.index == 0:
            camera.index = int(camera.index) + 1 # switches camera
        elif camera.index == 1:
            camera.index = int(camera.index) - 1
        else:
            camera.index = camera.index
        pass
    
    def img_preprocess(self, image_path, upload): # function to prep the image for the OCR
        img = cv2.imread(image_path) # loads the image passed as a parameter
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive_result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 38)
        #^uses adaptive thresholding to remove shadow and background noise           
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,5 ))
        morph_img = cv2.morphologyEx(adaptive_result, cv2.MORPH_CLOSE, kernel)# morphology to smooth out writing
        cv2.imwrite("crop.png", morph_img)
             # doesnt crop the img if it was uploaded
        morph_img = adaptive_result[224:705, 8:648]
        processed_img = cv2.imwrite(r"processed images\processed_img.png", morph_img)
        self.read_text_from_image(r"processed images\processed_img.png", morph_img) #reads the image         
   
    def read_text_from_image(self, proc_image_path, morph_img, language='en'):
        # Load the image using OpenCV
        image = cv2.imread(proc_image_path)
        # Initialize the OCR reader with the specified language
        reader = easyocr.Reader([language])
        # Perform OCR on the image
        result = reader.readtext(image)

        # Extract and print the recognized text
        recognized_text = ''.join([re.sub(r'^\W+|\W+$', '', text[1]) for text in result])
         #for debugging
         # passes string into the coefficient finder function
        print('Recognized Text: ' + recognized_text)
        recognized_text = recognized_text.replace('_', "-")
        recognized_text = recognized_text.replace(" ","")
        recognized_text = recognized_text.replace("/","1")
        recognized_text = recognized_text.replace("|","1")
        recognized_text = recognized_text.replace("l","1")
        recognized_text = recognized_text.replace("I","1")
        recognized_text = recognized_text.replace("O","0")
        recognized_text = recognized_text.replace("H","4")
        print('Recognized Text: ' + recognized_text)
         # remove confusion between underscores and minuses
        self.split_equation(recognized_text, morph_img) # passes equation into the split equation function

    def split_equation(self, equation, img): # separates both sides of the equation (before and after the =)
        equals_position = equation.find("=")
        equation = equation.upper() # easier to search from with constant case
        graph = False # variable to store whether the image is of a function or not
        img_valid = True
        if "Y" in equation:
            a, b, c = self.find_coefficients(equation.replace(" ", "")) # finds coefficients
            
            graph = True
        elif "=" in equation: # separates left and right side of equation
            left_hand_side = equation[:equals_position]
            right_hand_side = equation[(equals_position+1):]
            a1,b1,c1 = self.find_coefficients(left_hand_side.replace(" ",""))
            a2,b2,c2 = self.find_coefficients(right_hand_side.replace(" ",""))
            print(a1, b1, c1)
            a = a1-a2 # minuses right side from left side
            b = b1-b2
            c = c1-c2
        else:
            a, b, c = self.find_coefficients(equation)# if it's just an expression
            print("no equals present")
        print(a,b,c)
        if a != 0: # quadratic case
            self.solve_quadratic(a,b,c,graph, equation)
        elif a == 0 and b!=0:   #linear case
            self.solve_linear(b,c, graph, equation)
        else:  # invalid case
            title = "invalid image" 
            message = "please retake image"
            l = Login()
            l.show_alert_dialog(title, message) # shows error message
            img_valid = False # declares image as invalid
        if img_valid == True:
            cv2.imwrite(f"home images\{equation}.png", img) # adds to home screen images if image is valid
        list_of_images = os.listdir("home images") # uses os module to create a list of all the images in the directory
        if len(list_of_images) > 4: # ensures that directory can only have max 4 images to save computer resource
            files = glob.glob(os.path.join("home images", '*'))
            files.sort(key=os.path.getmtime)          
            os.remove(files[0])

    def find_coefficients(self, equation):
        # defines the x2 pattern
        pattern_x2 = re.compile(r'([-+]?\d*)x2\s*', re.IGNORECASE)
        # defines x pattern with negative lookahead to exclude x2
        pattern_x = re.compile(r'([-+]?\d+)\s*x(?!\d)', re.IGNORECASE)
        # pattern for constant terms
        pattern_constant = re.compile(r'([-+]?\d+)$')

        # Search for matches in the equation
        match_x2 = pattern_x2.search(equation)
        match_x = pattern_x.search(equation)
        match_constant = pattern_constant.search(equation)

        val_pattern_x2 = re.compile(r'x2|X2')
        #val_pattern_x = re.compile(r'x|X')
        x_is_there = re.search(r'[xX](?!2)', equation)
        x2_is_there = val_pattern_x2.search(equation)           
        # Set to 1 if x2 is present without explicit coefficient
        if bool(x2_is_there) == True:
            coefficient_x2 = int(match_x2.group(1)) if match_x2 and match_x2.group(1) else 1
        else:   
            coefficient_x2 = 0 # x2 = 0 if there aren't any pressent
    
        if bool(x_is_there) == True: # checks if x term is present
            coefficient_x = int(match_x.group(1)) if match_x and match_x.group(1) else 1
        else:
            coefficient_x = 0
        # Extract coefficients or set to 0 if not found
        constant_term = int(match_constant.group(1)) if match_constant and match_constant.group(1) else 0
        if equation == "x2":
            constant_term = 0
        #print(coefficient_x2, coefficient_x, constant_term)  
        return coefficient_x2, coefficient_x, constant_term        
    
    def solve_linear(self,b ,c, graph, equation): # solves the linear equation
        x = (-c)/b # computes the solution
        MDApp.get_running_app().root.current = 'linear' # changes to linear solution screen
        if graph == False: # different logic depending on whether the input is a graph or not
            ls = self.manager.get_screen('linear') #displays linear equation solution
            ls.solution(b,c,x)
        else:
            g = self.manager.get_screen('graph')
            g.plot_graph(0,b,c, "n/a", "n/a", equation, "Straight Line graph has 1 x-intercept") # plots linear graph

            
    def solve_quadratic(self, a, b, c, graph, equation): # calculates both exact and decimal versions of the root 
        discriminant1 = cmath.sqrt(b**2 - 4*a*c) # finds discriminant
        # Calculate the two solutions using the quadratic formula
        root1 = (-b + discriminant1) / (2*a)
        root2 = (-b - discriminant1) / (2*a)
        print(root1, " and ", root2) # prints the roots of the equation (debug)
        discriminant2 = b**2 - 4*a*c
        nature = '' # describes the nature of the equation
        if discriminant2 < 0: # if equation is complex
            nature = 'Equation is complex'
        elif discriminant2 == 0: # if equation has repeated roots
            nature = 'Equation has repeated roots'
        else: # if equation has 2 real roots
            nature = 'Equation has 2 real roots' 
        plus_minus_symbol = "\u00B1" # calls character from unicode character set
        root = "\u221A"      
        first_part = f"{-1*b}/{2*a}"
        second_part = f"{root}{discriminant2}/{2*a}"
        rounded_root1 = round(root1.real, 3) + round(root1.imag, 3) * 1j #for rounding
        rounded_root2 = round(root2.real, 3) + round(root2.imag, 3) * 1j
        dec_roots = f"Decimal roots: {rounded_root1}, {rounded_root2}" # decimal roots
        exact_roots = f"Exact roots: {first_part} {plus_minus_symbol} {second_part}" # fractional roots
        s = self.manager.get_screen('solve screen')
        if graph == False:
            s.solution(a,b,c,exact_roots, dec_roots, nature) # passes roots to the solve screen
            MDApp.get_running_app().root.current = 'solve screen'
        else:  
            g = self.manager.get_screen("graph") # go to plot class if the photo contains a function
            g.plot_graph(a,b,c, exact_roots, dec_roots, equation, nature)
        
        
    def on_enter(self): # function for adding images to the home screen widget
        # Specify the directory path where your images are stored
        image_directory = 'home images'
        self.ids.imagegrid.clear_widgets()
        # Iterate through the images in the directory
        for filename in os.listdir(image_directory):
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                image_path = os.path.join(image_directory, filename)
                file_name, _ = os.path.splitext(str(filename))
                # Create and add MDSmartTile for each image
                smart_tile = MDSmartTile(source=image_path)
                
                label = MDLabel(text=file_name, theme_text_color="Secondary", size_hint_y=None, height='30dp')
                smart_tile.add_widget(label)
                self.ids.imagegrid.add_widget(smart_tile) #adds the widget to the prespecified layout 

    def display_info(self): # displays the helpful information
        l = Login()
        message = "For best results write clearly and on a somewhat clear background. Capabilities are limited to graphing and solving quadratic and linear equations." 
        title = "Guide for Best Results"  
        l.show_alert_dialog(title, message)  # uses the show_alert_dialog function from the login class

    def account_screen(self, username):
        self.ids.topbar.title = "Welcome " + username
        self.ids.accountdetails.clear_widgets()  # clears previous widgets     
        
        password_panel = MDExpansionPanel(  # for change password functionality
        icon="icon.png", 
        content=ChangePasswordContent(),
        panel_cls=MDExpansionPanelOneLine(
        text="Change Password",
        ))  
        self.ids.accountdetails.add_widget(password_panel)   # adds both widgets to account screen   
        
    pass


class ChangePasswordContent(MDBoxLayout): # inherits MDBoxLayout
    def change_password(self):    
        l = Login()
        old_password = self.ids.old_password.text # grabs values from the text fields
        new_password = self.ids.new_password.text 
        with open('logindetails.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)    
            data =  list(csv_reader)
            for i, line in enumerate(data): # iterates through the file
                if line[0] == username and line[1] == old_password: # checks old password is correct                                     
                    data[i][1] = new_password # changes password in csv file
                    self.ids.old_password.text = ""
                    self.ids.new_password.text = ""
                    title = "Account Updated"
                    message = "Password has successfully been updated"
                    l.show_alert_dialog(title, message)
                elif line[0] == username and line[1] != old_password: # if passwords dont match, error message shown
                    title = "error"
                    message = "old password is incorrect"
                    l.show_alert_dialog(title, message)
        with open('logindetails.csv', mode='w', newline='') as csv_file: # writes new data into the updated csv
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(data)
    pass

class SIGNUP(MDScreen): # creates the sign up screen class which inherits MDScreen
    def create_account(self):
        new_username = self.ids.new_username.text # assigns text fields to the variables
        password1 = self.ids.password1.text
        password2 = self.ids.password2.text
        l = Login() # grabs the login class
        username_valid = True
        with open('logindetails.csv', mode='r') as old_file: # opens csv file
            csv_reader = csv.reader(old_file)
            for line in csv_reader:
                    if line[0] == new_username: # checks if username is taken
                        username_valid = False
        with open('logindetails.csv', mode='a', newline='') as csv_file: # open csv file in writer mode
            csv_writer = csv.writer(csv_file, delimiter=',') # sets formatting for the writer
            if username_valid == True and password1 == password2: # if the credentials are valid:
                csv_writer.writerow([new_username,password1]) # writes new credentials into the csv file
                self.ids.new_username.text = ""
                self.ids.password1.text = ""
                self.ids.password2.text = "" # clears the fields
                title = "Account Created"
                message = "Account has successfully been created"
                l.show_alert_dialog(title, message) # tells user that it was successful
            elif username_valid == False: # if username taken, error message shown
                title = "error"
                message = "Username is already taken"
                l.show_alert_dialog(title, message)
            else: # if passwords do not match, error message shown
                title = "error"
                message = "passwords do not match"
                l.show_alert_dialog(title, message)           
    pass

class solvescreen(MDScreen):# creates the solvescreen class which inherits MDScreen
    def solution(self, a,b,c, exact_solution, dec_solution, nature):
        print("debug2")
        # calling required unicode characters
        squared_symbol = "\u00B2"
        square_root_symbol = "\u221A"
        plus_minus_symbol = "\u00B1"
        # formatting output equation to avoid confusion
        real_coeff_x2 = a
        if a == 1:
            real_coeff_x2 = ""
        real_coeff_x = b
        if b == 1:
            real_coeff_x = ""
        real_const = c
        if c == 0:
            real_const = ""
        # creating equation
        self.ids.equation.text = str(real_coeff_x2) + f"x{squared_symbol} + " + str(real_coeff_x) + "x + " + str(real_const)
        
        
        # formatting of the solution
        self.ids.line1.text = f"-b{plus_minus_symbol}{square_root_symbol}b{squared_symbol}-4ac"
        self.ids.line1.secondary_text = "------------------"
        self.ids.line1.tertiary_text = f"2a"
        self.ids.line2.text = f"-{b}{plus_minus_symbol}{square_root_symbol}({b}{squared_symbol}-4x{a}x{c})"
        self.ids.line2.secondary_text = "------------------"
        self.ids.line2.tertiary_text = f"2x{a}"
        self.ids.line3.text = nature
        self.ids.line4.text = exact_solution
        self.ids.line5.text = dec_solution
    pass

class linearscreen(MDScreen): # screen that displays the linear solution
    def solution(self, b, c, solution):
        self.ids.method.text = f"{b}x = {(-1)*c}"
        self.ids.result.text = f"x = {solution}"
    pass

class graphscreen(MDScreen): 
    def plot_graph(self,a,b,c, exact_roots, dec_roots, equation, nature):
        MDApp.get_running_app().root.current = 'graph'
        self.ids.graph.clear_widgets()
        # Create a figure and axis for the graph
        fig, ax = plt.subplots()
        
        # Define the quadratic function (you can choose your own)
        x = np.linspace(-10, 10, 100)
        y = a*x**2 + b*x + c

        # Plot the quadratic function
        ax.plot(x, y, label=f"{equation}")
        ax.legend()

        # Add axis labels
        ax.axhline(0, color='black',linewidth=0.5)
        ax.axvline(0, color='black',linewidth=0.5)

        # Save the plot as an image
        plot_image_path = f'{equation}.png'
        fig.savefig(plot_image_path)
        plt.close(fig)

        # Create an Image widget with the plot
        plot_image = Image(source=f'{equation}.png')

        # Adds image to graph widget
        self.ids.graph.add_widget(plot_image)
        if exact_roots == "n/a": # if the graph is linear
            self.ids.exact.text = f"x-intercept = {str(-c/b)}"
            self.ids.nature.text = ""
            self.ids.decimal.text = ""
        else: # if the graph is quadratic
            self.ids.exact.text = exact_roots
            self.ids.decimal.text = dec_roots
            self.ids.nature.text = nature
    def delete_plot(self):
        pass

class myApp(MDApp): # creates the myApp class the inherits MDApp
    def build(self): # constructor method to create the application
        self.theme_cls.theme_style = "Dark" # formats the theme of the application
        self.theme_cls.primary_palette = "DeepOrange"
        Builder.load_file("Main.kv") # links the .kv file with the .py file
        sm = ScreenManager(transition = NoTransition()) # creates the screen manager for navigating between screen
        sm.add_widget(Login(name='login')) # defining the screens
        sm.add_widget(Main(name='main'))
        sm.add_widget(SIGNUP(name='signup'))
        sm.add_widget(solvescreen(name='solve screen'))
        sm.add_widget(graphscreen(name='graph'))
        sm.add_widget(linearscreen(name='linear'))
        return sm
    pass    

if __name__ == "__main__": # runs the app class when the program is run.
   myApp().run()
    