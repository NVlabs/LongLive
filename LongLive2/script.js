function isMobileDevice() {
    return /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

function isWeChat() {
    return window.navigator.userAgent.toLowerCase().includes('micromessenger');
}

function dismissWarning() {
    var warning = document.getElementById('mobile-warning');
    if (warning) warning.style.display = 'none';
}

if (isMobileDevice() || isWeChat()) {
    var warning = document.getElementById('mobile-warning');
    if (warning) warning.style.display = 'block';
}

document.querySelectorAll('.speed-controls').forEach(function(group) {
    group.addEventListener('click', function(event) {
        var button = event.target.closest('.speed-btn');
        if (!button) return;

        var speed = parseFloat(button.dataset.speed || '1');
        group.querySelectorAll('.speed-btn').forEach(function(item) {
            item.classList.remove('active');
        });
        button.classList.add('active');

        var section = group.closest('section');
        if (!section) return;
        section.querySelectorAll('video').forEach(function(video) {
            try {
                video.playbackRate = speed;
            } catch (error) {
                // Some browsers may reject playback-rate changes before metadata loads.
            }
        });
    });
});

(function attachShotPromptPopovers() {
    var promptSummaries = {
        african_savanna: [
            "An elephant walks across the savanna, partly hidden by an acacia tree.",
            "A lone elephant stands beneath an acacia tree, grazing or resting.",
            "The savanna stretches toward distant hills under a calm sky.",
            "A low angle frames the elephant's massive body and legs.",
            "The view studies the elephant's eye, ears, and shifting trunk.",
            "An elephant moves through open grassland beneath soft clouds.",
            "The elephant strides through tall dry grass, kicking up dust.",
            "A mature elephant stands in the savanna, ears gently flapping."
        ],
        aquamarine_underwater: [
            "A pufferfish and dolphin share a calm underwater moment.",
            "A dolphin swims upward through clear blue water with its mouth open.",
            "Golden ocean waves roll under a low sun at the surface.",
            "A whale rises from deep blue water toward the surface.",
            "A spotted dolphin glides gracefully through open water.",
            "A dolphin carries a small fish in its mouth while swimming.",
            "A dolphin shoots upward toward the shimmering surface.",
            "A sleek marine animal glides effortlessly through deep blue water."
        ],
        brown_bear_river: [
            "A wet brown bear lifts its head from the river and watches calmly.",
            "A bear wades from the water onto the riverbank.",
            "A bear moves behind foreground plants in a dense green habitat.",
            "The bear walks steadily from right to left across the ground.",
            "A close-up frames the bear's face as it looks forward.",
            "The bear forages near a mossy log, sniffing the ground.",
            "A bear passes behind ferns, moving quietly through the undergrowth.",
            "A bear walks away across sunlit ground, casting a clear shadow."
        ],
        cozy_red_room: [
            "A husky licks crumbs from its nose in a cozy red room.",
            "A fluffy dog plays on the carpet near a person in the living room.",
            "The dog moves to a doorway and receives a gentle pat.",
            "The dog looks up alertly in a sunlit hallway near the stairs.",
            "A husky follows the camera from the hallway into a bathroom.",
            "A hand places a treat on the husky's snout, and the dog licks it off.",
            "Two fluffy dogs hurry through the doorway, one following the other.",
            "A cream-colored dog trots down the hallway while a smaller dog appears nearby."
        ],
        frying_egg_closeup: [
            "An egg sizzles in a pan as bubbles form around its edges.",
            "A blackened wok heats on the stove, ready for cooking.",
            "Tomatoes, onions, and garlic simmer vigorously in a hot wok.",
            "A chef stirs bright vegetables in the wok with practiced hands.",
            "The cook keeps stirring as the sauce bubbles around the vegetables.",
            "The chef mixes eggs into the tomato sauce with a ladle.",
            "A glossy tomato-and-egg stir-fry fills the frame, finished with scallions."
        ],
        indoor_grooming_scene: [
            "A groomer trims fur around the dog's paw on a grooming table.",
            "She guides electric clippers carefully through the dog's coat.",
            "The dog stays calm as the groomer works close to its face.",
            "The groomer holds the dog's ear steady and trims with scissors.",
            "A dog with a purple accessory waits among grooming tools.",
            "The groomer and dog share a quiet, tender moment during the session."
        ],
        sample_sequence_0062_0: [
            "An office worker reviews printed charts at a tidy desk before a meeting.",
            "The office worker spots a typo and taps the mistake with a pen.",
            "The office worker studies the paperwork while blinds stripe the quiet desk.",
            "The office worker fixes a jammed stapler with a paperclip.",
            "The office worker straightens the corrected pages into a slim folder.",
            "The office worker smooths a blouse, checks the list, and switches off the lamp."
        ],
        sandy_beach_driftwood: [
            "A woman walks across a bright sandy beach past driftwood and rocks.",
            "She presents a freshly caught fish on a sunny beach.",
            "Hands inspect and rotate a large red-orange fish on the beach.",
            "The woman holds up her catch, then arranges driftwood for a fire.",
            "A beach campfire smolders as the woman tends it and shows the fish.",
            "A hand adjusts driftwood while flames flicker in the sand.",
            "The woman kneels beside a driftwood fire setup, preparing to light it.",
            "A small beach fire burns while the woman sits facing the ocean."
        ],
        shimmering_puzzle_surface: [
            "A holiday figure kneels by scattered green fragments, gathering and inspecting them.",
            "The holiday figure tumbles backward while taking a stressful phone call.",
            "The holiday figure holds a phone to his ear and reacts with weary concern.",
            "The holiday figure listens on the phone, turning his head with surprise.",
            "The holiday figure dances as children run past decorated Christmas trees.",
            "A festive trio walks forward between rows of decorated holiday trees."
        ],
        stonehenge_blue_sky: [
            "The view pans up and across massive Stonehenge stones beneath a blue sky.",
            "The camera glides forward through sunlit Stonehenge arches.",
            "Two men gesture and converse among foggy Stonehenge stones.",
            "Stonehenge stands in open grass as the view explores the ring.",
            "A ground-level view rises to reveal Stonehenge's towering scale.",
            "A man presents the monument while another listens before the stones.",
            "A guide gestures animatedly while his companion listens among the megaliths.",
            "A rendered camera orbit circles a prehistoric stone ring."
        ],
        sunlit_balcony_tour: [
            "A bearded presenter walks along a balcony and stops near a black door.",
            "He opens a glass door, enters the bright interior, and gestures to the camera.",
            "He presents a high-end living area, gesturing toward the furniture and door.",
            "He opens a dark wooden door and walks into a sleek room.",
            "He demonstrates built-in cabinetry while explaining the storage design.",
            "He points upward to a hidden cabinet feature in the modern interior."
        ],
        vintage_archival_scene: [
            "Crewmen move around a vintage military helicopter as one climbs from the side door.",
            "A shirtless mechanic works inside a cramped engine bay among exposed machinery.",
            "A metal rotor spins slowly against a pale overcast sky.",
            "A helicopter lifts from a naval deck while personnel monitor the takeoff.",
            "Helicopters fly in formation over hazy wilderness and rivers.",
            "A helmeted pilot flies from a grainy cockpit as another helicopter passes outside.",
            "A helicopter crosses misty lowland from an aircraft-window perspective."
        ],
        warm_indoor_dining: [
            "A spread of porcelain bowls presents fried bites, herbs, sauce, and shared dishes.",
            "Plates of marinated meat and crisp dumplings sit ready for the meal.",
            "Chopsticks lift glossy noodles from a steaming bowl of rich broth.",
            "Sliced beef rises from the bowl, dripping sauce and scallions.",
            "Chopsticks pull noodles through red broth with greens and sesame.",
            "Thick noodles are lifted high from the soup in a close-up.",
            "Chopsticks pick up a sauced beef slice with peanuts and scallions.",
            "Chopsticks lift glossy noodles from a dark-sauced dish on the table."
        ],
        robot_demo: [
            "WALL-E moves through a warm, cluttered workshop and notices a tiny insect.",
            "WALL-E gently studies a small green plant in soft golden light.",
            "The camera pushes in on a fragile sprout emerging from rich soil.",
            "Two weathered robotic hands slowly gather around the tiny seedling.",
            "A robotic hand carefully places the sprout into a small container."
        ],
        robot_demo_nvfp4: [
            "WALL-E moves through a warm, cluttered workshop and notices a tiny insect.",
            "WALL-E gently studies a small green plant in soft golden light.",
            "The camera pushes in on a fragile sprout emerging from rich soil.",
            "Two weathered robotic hands slowly gather around the tiny seedling.",
            "A robotic hand carefully places the sprout into a small container."
        ]
    };

    function getFileName(src) {
        var clean = String(src || '').split('?')[0].split('#')[0];
        try {
            clean = decodeURIComponent(clean);
        } catch (error) {}
        return clean.split('/').pop() || '';
    }

    function getPromptKeyAndShot(src) {
        var name = getFileName(src)
            .replace(/(?:\.mp4)+$/i, '')
            .replace(/^【[^】]+】/g, '')
            .replace(/[（(]\s*TODO\s*[)）]/ig, '')
            .trim()
            .replace(/\s+/g, '_');

        if (/openai_multishot_0062_0/i.test(name)) {
            return { key: 'sample_sequence_0062_0' };
        }

        var lower = name.toLowerCase();
        lower = lower.replace(/^(bf16|nvfp4)_/, '');

        if (lower === 'robot_demo' || lower === 'robot_demo_nvfp4') {
            return { key: lower };
        }

        var shotMatch = lower.match(/_(\d+)$/);
        if (shotMatch) {
            lower = lower.replace(/_\d+$/, '');
        }

        return { key: lower };
    }

    function getPromptsForVideo(video) {
        var source = video.querySelector('source');
        var src = source ? source.getAttribute('src') : video.getAttribute('src');
        var match = getPromptKeyAndShot(src);
        var summaries = promptSummaries[match.key];
        return Array.isArray(summaries) ? summaries.filter(Boolean) : [];
    }

    function createShotPromptPopover(prompts) {
        var popover = document.createElement('div');
        var title = document.createElement('div');
        var list = document.createElement('ol');

        popover.className = 'caption-popover';
        title.className = 'caption-popover-title';
        list.className = 'caption-list';
        title.textContent = 'Shot Prompts';

        prompts.forEach(function(prompt, index) {
            var item = document.createElement('li');
            var shotLabel = document.createElement('span');
            var promptText = document.createElement('span');

            shotLabel.className = 'caption-shot';
            promptText.className = 'caption-text';
            shotLabel.textContent = 'Shot ' + (index + 1);
            promptText.textContent = prompt;

            item.appendChild(shotLabel);
            item.appendChild(promptText);
            list.appendChild(item);
        });

        popover.appendChild(title);
        popover.appendChild(list);
        return popover;
    }

    document.querySelectorAll('.demo-card').forEach(function(card) {
        if (card.querySelector('.caption-popover')) return;

        var video = card.querySelector('video');
        if (!video) return;

        var prompts = getPromptsForVideo(video);
        if (!prompts.length) return;

        card.classList.add('caption-card');
        card.appendChild(createShotPromptPopover(prompts));
        card.tabIndex = card.tabIndex >= 0 ? card.tabIndex : 0;
        card.setAttribute('aria-label', 'Demo video with shot prompts');
    });
})();
