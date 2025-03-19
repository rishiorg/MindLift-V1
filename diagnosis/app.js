// Constructor function for question objects.
function Question(question, questionCategory, responseOptions, responseOptionScores) {
    this.question = question;                   // string
    this.questionCategory = questionCategory;   // string
    this.responseOptions = responseOptions;     // array of strings
    this.responseOptionScores = responseOptionScores;   // array of numbers
    this.userResponse;                          // index in responseOptions
}

// Function to display one question on webpage.
function askQuestion(questionObject) {

    // Clear page.
    let bannerEl = document.getElementById("banner-image");
    bannerEl.innerText = "";
    let mainEl = document.getElementsByTagName("main")[0];
    mainEl.innerText = "";

    // Add question title to page.
    let questionTitle = document.createElement("h1");
    for (let i = 0; i < questionObjects.length; i++) {
        if (questionObjects[i].question == questionObject.question) {
            questionTitle.innerText = `Question ${i+1} of ${questionObjects.length}`;
            break;
        }
    }
    mainEl.appendChild(questionTitle);

    // Display question on page.
    let questionEl = document.createElement("p");
    questionEl.innerText = questionObject.question;
    questionEl.setAttribute("id", "question");
    mainEl.appendChild(questionEl);

    // Display answer choices on page.
    let answersEl = document.createElement("ul");
    for (i = 0; i < questionObject.responseOptions.length; i++) {
        let answerEl = document.createElement("li");
        answerEl.innerText = questionObject.responseOptions[i];
        answersEl.appendChild(answerEl);
    }
    mainEl.appendChild(answersEl);

    document.getElementsByTagName("ul")[0].addEventListener("click", handleResponse);   // Wait for user to click their response choice.
}

// Function to handle user response (click).
// Records which response option the user selected and asks the next question if there is a next question to be asked, otherwise shows results.
function handleResponse(event) {
    let elementClicked = event.target; // returns the list element "<li>3</li>" that was clicked on
    
    // Record user selection.
    let questionObject;

    // Find the questionObject that corresponds to the question the user responded to.
    for (i = 0; i < questionObjects.length; i++) {
        if (document.getElementsByTagName("p")[0].innerText == questionObjects[i].question) {
            questionObjects[i].userResponse = Array.from(elementClicked.parentNode.children).indexOf(elementClicked);
            break;
        }
    }
    // Ask next question if questions are left to be asked. Otherwise, get results.
    if (questionsAlreadyAsked.length == questionObjects.length) {
        document.getElementsByTagName("ul")[0].removeEventListener("click", handleResponse);
        getResults();
    } else {
        askNextQuestion();
    }
}

// Shows the next question on the page. Keeps track of the question that it asks in questionsAlreadyAsked.
function askNextQuestion() {

    // Ask all the questions in questionObjects. Do not ask repeat questions.
    for (let i = 0; i < questionObjects.length; i++) {

        // if questoinObjects[i] is in questionsAlreadyAsked, go to next questionObject
        if (questionsAlreadyAsked.includes(questionObjects[i])) {
            // do nothing, go to next question.

        // else ask the question and push it into the questions already asked array.
        } else {
            questionsAlreadyAsked.push(questionObjects[i]); // Keep track of which questions the user has been asked.
            askQuestion(questionObjects[i]);
            break;
        }
    }
}
 
// Calculates results of user responses into format results chart can use.
function getResults() {

    // For loop to loop through questionObjects.
    // After this loop completes should have array that looks like this: ["Depression", "GAD", "SAD", "ADHD", "PTSD", "GD"];
    let questionCategories = [];
    let sumOfEachCategory = [];
    for (let i = 0; i < questionObjects.length; i++) {
        // Push new questionCategories into an array to keep track of which questionCategories exist
        if (questionCategories.includes(questionObjects[i].questionCategory)) {
            // Question category is already in the questionCategories list, so do nothing.
        } else {
            questionCategories.push(questionObjects[i].questionCategory)
            sumOfEachCategory.push(0);
        }
    }

    // New array to keep track of total values for each category: sumOfEachCategory = [sumOfDepressionScores, sumofGADScores, ...]
    // For loop to assemble the results for each index in the question categories array.
    for (let j = 0; j < questionObjects.length; j++) {
        for (let i = 0; i < questionCategories.length; i++) {
            if (questionCategories[i] == questionObjects[j].questionCategory) {
                sumOfEachCategory[i] = sumOfEachCategory[i] + questionObjects[j].responseOptionScores[Number(questionObjects[j].userResponse)];
            }
        }
    }

    showResults(questionCategories, sumOfEachCategory)
}

// Displays results chart and mental health descriptions.
function showResults(x, y) {
    
    // Adds results chart to webpage.
    let mainEl = document.getElementsByTagName("main")[0];
    mainEl.innerText = "";

    // Add title to page.
    let titleEl = document.createElement("h1");
    titleEl.innerText = "Your Results";
    titleEl.setAttribute("id", "results-h1");
    mainEl.appendChild(titleEl);

    // Put canvas in a div container to be able to resize with css easily.
    let canvasContainer = document.createElement("div");
    canvasContainer.setAttribute("class","chart-divs");
    mainEl.appendChild(canvasContainer);

    // Create canvas to put chart onto.
    let canvasEl = document.createElement("canvas");
    canvasContainer.appendChild(canvasEl);
    let ctx = canvasEl.getContext("2d");
    
    // Create chart.
    var myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: x,
            datasets: [{
                label: 'Score',
                data: y,
                backgroundColor: '#6B88D1'
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: '% of Maximum Possible Score'
                    }
                }
            },
        }
    });

    // Add references/attributions for sources used.
    let creditEl = document.createElement("p");
    creditEl.setAttribute("id","credit-p");
    creditEl.innerText = `The questions in this quiz are sourced from the PHQ-2, PHQ-9, GAD-2 (all copyright Pfizer, no permission required to reproduce or distribute), Adult ADHD Self Report Scale V1.1 (copyright World Health Organization, use allowed with acknowledgement of copyright), and Utrecht Gender Dysphoria Scale - Gender Spectrum (copyrighted University of Minnesota, license for GALA expires Oct 5, 2024 under ID 6TJ7).`;
    mainEl.appendChild(creditEl);

    // Condition description information is stored in this object.
    let descriptionObjects = {
        conditions: ['Depression','Generalized Anxiety','Social Anxiety','ADHD','Gender Dysphoria','PTSD'],
        links: [
            'https://www.nimh.nih.gov/health/topics/depression',
            'https://www.nimh.nih.gov/health/publications/generalized-anxiety-disorder-gad',
            'https://www.nimh.nih.gov/health/statistics/social-anxiety-disorder',
            'https://www.nimh.nih.gov/health/topics/attention-deficit-hyperactivity-disorder-adhd',
            'https://en.wikipedia.org/wiki/Gender_dysphoria',
            'https://www.nimh.nih.gov/health/topics/post-traumatic-stress-disorder-ptsd'
        ],
        descriptions: [
            `Depression (major depressive disorder or clinical depression) is a common but serious mood disorder. It causes severe symptoms that affect how you feel, think, and handle daily activities, such as sleeping, eating, or working.`,
            `Occasional anxiety is a normal part of life. You might worry about things like health, money, or family problems. But people with generalized anxiety disorder (GAD) feel extremely worried or feel nervous about these and other things—even when there is little or no reason to worry about them. People with GAD find it difficult to control their anxiety and stay focused on daily tasks.`,
            `Social anxiety disorder (formerly social phobia) is characterized by persistent fear of one or more social or performance situations in which the person is exposed to unfamiliar people or to possible scrutiny by others. The individual fears that he or she will act in a way (or show anxiety symptoms) that will be embarrassing and humiliating.`,
            `Attention-deficit/hyperactivity disorder (ADHD) is marked by an ongoing pattern of inattention and/or hyperactivity-impulsivity that interferes with functioning or development. [...] Some people with ADHD mainly have symptoms of inattention. Others mostly have symptoms of hyperactivity-impulsivity. Some people have both types of symptoms.`,
            `Gender dysphoria (GD) is the distress a person feels due to a mismatch between their gender identity—their personal sense of their own gender—and their sex assigned at birth.`,
            `Post-traumatic stress disorder (PTSD) is a disorder that develops in some people who have experienced a shocking, scary, or dangerous event. It is natural to feel afraid during and after a traumatic situation. Fear triggers many split-second changes in the body to help defend against danger or to avoid it.`
        ],
        source: [
            '—National Institute of Mental Health',
            '—National Institute of Mental Health',
            '—National Institute of Mental Health',
            '—National Institute of Mental Health',
            '—Wikipedia',
            '—National Institute of Mental Health'
        ]
    }

    // Add condition descriptions contained in discriptionObjects to page.
    let figureContainer = document.createElement("div");
    figureContainer.setAttribute("id", "figure-container");

    for (i = 0; i < descriptionObjects.conditions.length; i++) {
        let definitionFig = document.createElement("figure");
        figureContainer.appendChild(definitionFig);
    
        let link = document.createElement("figcaption");
        link.innerText = descriptionObjects.conditions[i];
        definitionFig.appendChild(link);
    
        let quote = document.createElement("blockquote");
        quote.innerText = descriptionObjects.descriptions[i];
        definitionFig.appendChild(quote);
    
        let source = document.createElement("a");
        source.innerText = descriptionObjects.source[i];
        source.href = descriptionObjects.links[i];
        definitionFig.appendChild(source);
    }

    mainEl.appendChild(figureContainer);

    // Add a "Retake Assessment" button
    let retakeButton = document.createElement("button");
    retakeButton.innerText = "Retake Assessment";
    retakeButton.setAttribute("class", "button-3d");
    retakeButton.addEventListener("click", resetAssessment);
    mainEl.appendChild(retakeButton);
    
    // Add a "Print Results" button
    let printButton = document.createElement("button");
    printButton.innerText = "Print Results";
    printButton.setAttribute("class", "button-3d");
    printButton.addEventListener("click", function() {
        window.print();
    });
    mainEl.appendChild(printButton);
}

// Reset assessment to start again
function resetAssessment() {
    // Reset all user responses
    for (let i = 0; i < questionObjects.length; i++) {
        questionObjects[i].userResponse = undefined;
    }
    
    // Clear the questions already asked array
    questionsAlreadyAsked = [];
    
    // Return to landing page
    showLandingPage();
}

// Show the landing page
function showLandingPage() {
    // Clear main content
    let mainEl = document.getElementsByTagName("main")[0];
    mainEl.innerText = "";
    
    // Get the banner element
    let bannerEl = document.getElementById("banner-image");
    bannerEl.innerText = "";
    
    // Recreate the landing page structure
    let section = document.createElement("section");
    section.setAttribute("id", "quiz-box");
    section.setAttribute("class", "glass-effect floating-element");
    
    let quizBoxText = document.createElement("div");
    quizBoxText.setAttribute("id", "quiz-box-text");
    
    let h1 = document.createElement("h1");
    h1.innerText = "Mental Health Assessment";
    
    let h2 = document.createElement("h2");
    h2.innerText = "Would you like to take a comprehensive mental health screening?";
    
    let p = document.createElement("p");
    p.setAttribute("class", "description");
    p.innerText = "This assessment uses validated screening tools to help you understand potential mental health concerns. All responses are confidential and processed only in your browser.";
    
    quizBoxText.appendChild(h1);
    quizBoxText.appendChild(h2);
    quizBoxText.appendChild(p);
    
    let button = document.createElement("button");
    button.setAttribute("class", "button-3d");
    button.setAttribute("id", "yes");
    
    let span = document.createElement("span");
    span.innerText = "Begin Assessment";
    
    let buttonEffect = document.createElement("div");
    buttonEffect.setAttribute("class", "button-effect");
    
    button.appendChild(span);
    button.appendChild(buttonEffect);
    
    section.appendChild(quizBoxText);
    section.appendChild(button);
    
    bannerEl.appendChild(section);
    
    // Re-add event listener
    document.getElementById("yes").addEventListener("click", askNextQuestion);
}

// Question bank and associated arrays to build question objects out of.
QuestionCategory = ['Depression','Depression','Depression','Generalized Anxiety','Generalized Anxiety','Social Anxiety','Social Anxiety','Social Anxiety','Social Anxiety','Social Anxiety','ADHD','ADHD','ADHD','ADHD','ADHD','ADHD','Gender Dysphoria','Gender Dysphoria','Gender Dysphoria','Gender Dysphoria','Gender Dysphoria','PTSD','PTSD','PTSD']
Questions = ['Little interest or pleasure in doing things','Feeling down, depressed, or hopeless','Thoughts that you would be better off dead, or thoughts of hurting yourself in some way?','Feeling nervous, anxious, or on edge','Not being able to stop or control worrying','I find it difficult to mix comfortably with the people I work with.','I am at ease meeting people at parties, etc.','I have difficulty talking with other people.','I worry about expressing myself in case I appear awkward.','I find myself worrying that I won\'t know what to say in social situations.','How often do you have trouble wrapping up the final details of a project, once the challenging parts have been done?','How often do you have difficulty getting things in order when you have to do a task that requires organization?','How often do you have problems remembering appointments or obligations?','When you have a task that requires a lot of thought, how often do you avoid or delay getting started?','How often do you fidget or squirm with your hands or feet when you have to sit down for a long time?','How often do you feel overly active and compelled to do things, like you were driven by a motor?','Every time someone treats me like my assigned sex I feel hurt','It feels good to live as my affirmed gender','A life in my affirmed gender is more attractive for me than a life in my assigned sex','I wish I had been born as my affirmed gender','I feel unhappy when someone misgenders me','Memories of trauma come to mind at unwanted moments','At random times, I feel a burst of anger and/or guilt about past trauma','I feel a physical effect when remembering a past trauma, effects can range from racing pulse, excessive sweating, chills or shaking?']
ResponseOptions = [['Not at all','Several days','More than half the days','Nearly every day'],['Not at all','Several days','More than half the days','Nearly every day'],['Not at all','Several days','More than half the days','Nearly every day'],['Not at all','Several days','More than half the days','Nearly every day'],['Not at all','Several days','More than half the days','Nearly every day'],['Not at all','Slightly','Moderately','Very','Extremely'],['Not at all','Slightly','Moderately','Very','Extremely'],['Not at all','Slightly','Moderately','Very','Extremely'],['Not at all','Slightly','Moderately','Very','Extremely'],['Not at all','Slightly','Moderately','Very','Extremely'],['Never','Rarely','Sometimes','Often','Very Often'],['Never','Rarely','Sometimes','Often','Very Often'],['Never','Rarely','Sometimes','Often','Very Often'],['Never','Rarely','Sometimes','Often','Very Often'],['Never','Rarely','Sometimes','Often','Very Often'],['Never','Rarely','Sometimes','Often','Very Often'],['Disagree completely','Disagree','Neither agree nor disagree','Agree','Agree completely'],['Disagree completely','Disagree','Neither agree nor disagree','Agree','Agree completely'],['Disagree completely','Disagree','Neither agree nor disagree','Agree','Agree completely'],['Disagree completely','Disagree','Neither agree nor disagree','Agree','Agree completely'],['Disagree completely','Disagree','Neither agree nor disagree','Agree','Agree completely'],['Disagree completely','Disagree','Neither agree nor disagree','Agree','Agree completely'],['Disagree completely','Disagree','Neither agree nor disagree','Agree','Agree completely'],['Disagree completely','Disagree','Neither agree nor disagree','Agree','Agree completely']]
ScoringKey = [[0,11.11,22.22,33.33],[0,11.11,22.22,33.33],[0,11.11,22.22,33.33],[0,16.67,33.33,50],[0,16.67,33.33,50],[0,5,10,15,20],[20,15,10,5,0],[0,5,10,15,20],[0,5,10,15,20],[0,5,10,15,20],[0,0,16.67,16.67,16.67],[0,0,16.67,16.67,16.67],[0,0,16.67,16.67,16.67],[0,0,0,16.67,16.67],[0,0,0,16.67,16.67],[0,0,0,16.67,16.67],[0,0,0,10,20],[0,0,0,10,20],[0,0,0,10,20],[0,0,0,10,20],[0,0,0,10,20],[0,0,0,16.67,33.33],[0,0,0,16.67,33.33],[0,0,0,16.67,33.33]]

let questionObjects = [];

for (let i = 0; i < Questions.length; i++) {
    let tempObj = new Question(Questions[i], QuestionCategory[i], ResponseOptions[i], ScoringKey[i])
    questionObjects.push(tempObj)
}

// Keep track of which questions have already been asked.
let questionsAlreadyAsked = [];

// Function to handle the progress bar updates
function updateProgressBar(currentQuestion, totalQuestions) {
    // Create or update progress bar
    let progressContainer = document.getElementById('progress-container');
    
    if (!progressContainer) {
        progressContainer = document.createElement('div');
        progressContainer.setAttribute('id', 'progress-container');
        progressContainer.setAttribute('class', 'progress-container');
        
        let progressBar = document.createElement('div');
        progressBar.setAttribute('id', 'progress-bar');
        progressBar.setAttribute('class', 'progress-bar');
        
        let progressText = document.createElement('span');
        progressText.setAttribute('id', 'progress-text');
        
        progressContainer.appendChild(progressBar);
        progressContainer.appendChild(progressText);
        
        let mainEl = document.getElementsByTagName('main')[0];
        mainEl.insertBefore(progressContainer, mainEl.firstChild);
    }
    
    // Update progress
    let progress = (currentQuestion / totalQuestions) * 100;
    document.getElementById('progress-bar').style.width = progress + '%';
    document.getElementById('progress-text').innerText = `Question ${currentQuestion} of ${totalQuestions}`;
}

// Function to initialize the application
function initApp() {
    // Apply smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Add responsive behavior for navigation
    const handleResize = () => {
        if (window.innerWidth < 768) {
            document.querySelector('nav').classList.add('mobile-nav');
        } else {
            document.querySelector('nav').classList.remove('mobile-nav');
        }
    };
    
    // Call on load and add event listener
    handleResize();
    window.addEventListener('resize', handleResize);
    
    // Add click event for the start button
    document.getElementById("yes").addEventListener("click", askNextQuestion);
    
    // Add accessibility improvements
    document.querySelectorAll('button, a').forEach(element => {
        if (!element.hasAttribute('aria-label') && element.innerText) {
            element.setAttribute('aria-label', element.innerText);
        }
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);

// Handle warning for suicide questions
function displaySuicideWarning(callback) {
    // Create modal container
    let modalContainer = document.createElement('div');
    modalContainer.setAttribute('id', 'modal-container');
    modalContainer.setAttribute('class', 'modal-container');
    
    // Create modal content
    let modal = document.createElement('div');
    modal.setAttribute('class', 'modal glass-effect');
    
    let heading = document.createElement('h2');
    heading.innerText = 'Important Information';
    
    let content = document.createElement('p');
    content.innerText = 'If you are experiencing thoughts of self-harm or suicide, please contact a mental health professional or crisis line immediately:';
    
    let crisisInfo = document.createElement('p');
    crisisInfo.innerHTML = '<strong>988 Suicide & Crisis Lifeline:</strong> Call or text 988, or chat at 988lifeline.org';
    
    let buttonContainer = document.createElement('div');
    buttonContainer.setAttribute('class', 'modal-buttons');
    
    let continueButton = document.createElement('button');
    continueButton.setAttribute('class', 'button-3d');
    continueButton.innerText = 'Continue with Assessment';
    
    let helpButton = document.createElement('button');
    helpButton.setAttribute('class', 'button-3d help-button');
    helpButton.innerText = 'Get Help Now';
    
    // Append all elements
    buttonContainer.appendChild(continueButton);
    buttonContainer.appendChild(helpButton);
    
    modal.appendChild(heading);
    modal.appendChild(content);
    modal.appendChild(crisisInfo);
    modal.appendChild(buttonContainer);
    
    modalContainer.appendChild(modal);
    document.body.appendChild(modalContainer);
    
    // Add event listeners
    continueButton.addEventListener('click', function() {
        document.body.removeChild(modalContainer);
        if (callback) callback();
    });
    
    helpButton.addEventListener('click', function() {
        window.open('https://988lifeline.org', '_blank');
    });
}

// Check for suicide-related questions and display warning if needed
function checkSuicideContent(question) {
    const suicideKeywords = ['suicide', 'kill', 'hurt yourself', 'better off dead'];
    
    if (suicideKeywords.some(keyword => question.toLowerCase().includes(keyword))) {
        return true;
    }
    return false;
}

// Override askQuestion to include suicide warning check
const originalAskQuestion = askQuestion;
askQuestion = function(questionObject) {
    if (checkSuicideContent(questionObject.question)) {
        displaySuicideWarning(() => {
            originalAskQuestion(questionObject);
        });
    } else {
        originalAskQuestion(questionObject);
    }
};